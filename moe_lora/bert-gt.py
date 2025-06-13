import os
import types
import torch
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5DenseActDense
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# hyperparams
k = 20
G = 8               # 专家组数（可和 config.split_num 对齐）
r = 4               # LoRA rank

tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()

sst2 = load_dataset('sst2')
sst2_dev = sst2['validation']


def prepare_prototype_lora(ffn_self):
    """
    在第一次打补丁时调用一次，基于 ffn_self.wi.weight 和 ffn_self.patterns
    聚类成 G 个组，计算每组 prototype 和每个专家的 LoRA 差量。
    """
    # 1) 拿到 wi.weight (d_h, d_ff)
    W = ffn_self.wi.weight.detach().clone()  # (d_hidden, d_ffn)
    d_h, d_ff = W.shape

    # 2) 提取 E 个专家的特征向量（这里用每行平均做演示）
    #    ffn_self.patterns: (E, d_ff) bool，标记每个专家覆盖哪些维度
    patterns = ffn_self.patterns.bool().cpu().numpy()
    E = patterns.shape[0]
    feats = []
    for e in range(E):
        cols = patterns[e]                # bool mask for d_ff dims
        subW = W[:, cols]               # (d_h, #cols)
        feats.append(subW.mean(dim=1).cpu().numpy())
    feats = np.stack(feats, axis=0)     # (E, d_h)

    # 3) KMeans to G groups
    feats_norm = normalize(feats, axis=1)
    km = KMeans(n_clusters=G, random_state=0).fit(feats_norm)
    labels = km.labels_               # (E,)
    ffn_self.expert2group = labels

    # 4) compute prototypes per group
    proto = torch.zeros((G, d_h, d_ff), device=W.device)
    counts = torch.zeros(G, device=W.device)
    for e, g in enumerate(labels):
        proto[g] += W
        counts[g] += 1
    proto = proto / counts.view(G, 1, 1)
    # store
    ffn_self.prototypes = proto       # (G, d_h, d_ff)

    # 5) compute LoRA A,B for each expert
    ffn_self.lora_A = {}
    ffn_self.lora_B = {}
    for e in range(E):
        g = labels[e]
        Delta = W - proto[g]           # full-matrix Δ; patterns mask will zero out irrelevant dims in forward
        # low-rank SVD
        U, S, Vt = torch.linalg.svd(Delta, full_matrices=False)
        U_r = U[:, :r]                 # (d_h, r)
        S_r = S[:r].sqrt()             # (r,)
        Vt_r = Vt[:r, :]               # (r, d_ff)
        A_e = U_r * S_r.unsqueeze(0)   # (d_h, r)
        B_e = S_r.unsqueeze(1) * Vt_r  # (r, d_ff)
        ffn_self.lora_A[e] = A_e
        ffn_self.lora_B[e] = B_e


def _forward_with_protolora(ffn_self, hidden_states):
    """
    Prototype + LoRA 差量前向：
      1) 用组原型做一次 dense 投影
      2) 用 LoRA A,B 叠加低秩增量
      3) ReLU → 掩码 → dropout → sparse wo
    """
    B, L, d_h = hidden_states.shape
    d_ff = ffn_self.prototypes.shape[-1]

    # 1) standard routing mask
    flat = hidden_states.view(-1, d_h)
    flat_norm = flat / flat.norm(dim=-1, keepdim=True)
    scores = ffn_self.mlp(flat_norm)              # (B*L, E)
    topk = torch.topk(scores, k=k, dim=-1)[1]     # (B*L, k)
    mask = torch.nn.functional.embedding(topk,    # (B*L, k, d_ff)
              ffn_self.patterns).sum(dim=1)       # sum over k -> (B*L, d_ff)
    mask = mask.view(B, L, d_ff).bool()           # (B, L, d_ff)

    # 2) prototype projection
    # 找到当前 ffn_self 属于哪个组 (每层只有一个组原型？也可选用多组)
    # 这里我们简单假设整个模块都用第一个组原型
    proto_W = ffn_self.prototypes[0]              # (d_h, d_ff)
    H = hidden_states @ proto_W                   # (B, L, d_ff)

    # 3) LoRA 差量
    # 扁平后批量叠加
    topk_flat = topk.view(-1)
    A = torch.stack([ffn_self.lora_A[e] for e in topk_flat], dim=0)   # (B*L*k, d_h, r)
    Bm = torch.stack([ffn_self.lora_B[e] for e in topk_flat], dim=0)  # (B*L*k, r, d_ff)

    hs_flat = hidden_states.view(-1, d_h).unsqueeze(1).expand(-1, k, -1)
    hs_flat = hs_flat.reshape(-1, d_h)                              # (B*L*k, d_h)

    delta = (hs_flat @ A).bmm(Bm)                                    # (B*L*k, d_ff)
    delta = delta.view(B, L, k, d_ff).sum(dim=2)                     # (B, L, d_ff)

    H = H + delta

    # 4) activation + mask + dropout + sparse wo
    H = ffn_self.act(H)
    H = H.masked_fill(~mask, 0.0)
    H = ffn_self.dropout(H)
    return ffn_self.wo(H)


def change_forward(model, k=20):
    # patch every DenseReluDense
    for layer_idx, layer in enumerate(model.encoder.block):
        ffn = layer.layer[1].DenseReluDense
        # 先准备 prototype+LoRA
        prepare_prototype_lora(ffn)
        # 然后替换前向
        ffn.forward = types.MethodType(_forward_with_protolora, ffn)

    for layer_idx, layer in enumerate(model.decoder.block):
        ffn = layer.layer[2].DenseReluDense
        prepare_prototype_lora(ffn)
        ffn.forward = types.MethodType(_forward_with_protolora, ffn)


# apply patch
change_forward(model, k)

# eval sst2
pred = []
for inst in sst2_dev:
    input_ids = tokenizer("sst2 sentence: "+inst['sentence'], return_tensors="pt").input_ids.cuda()
    dec_input_ids = tokenizer("<extra_id_0>", return_tensors="pt").input_ids.cuda()[:, :1]
    out = model(input_ids=input_ids, labels=dec_input_ids)
    # 假设 1465,2841 对应标签
    pred.append(int(out.logits[:,0,1465]>out.logits[:,0,2841])==inst['label'])

print("Acc", sum(pred)/len(pred), 'k', k)
