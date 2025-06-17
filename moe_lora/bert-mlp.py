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
k = 20    # 每个 token 激活的专家数
G = 8     # 将 128 个 expert 分成 8 组
r = 4     # LoRA 差量的秩

# 加载模型与数据
tokenizer = T5Tokenizer.from_pretrained('t5-base')
config    = T5Config.from_pretrained('t5-base')
model     = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()

sst2      = load_dataset('sst2')
sst2_dev  = sst2['validation']


def prepare_prototype_lora(ffn_self):
    """
    仅调用一次，基于 ffn_self.wi.weight 和 ffn_self.patterns
    聚类成 G 组，计算每组 prototype 和每个 expert 的 LoRA 差量。
    """
    # 1) 拷贝 wi.weight
    W = ffn_self.wi.weight.detach().clone()   # (d_h, d_ff)
    d_h, d_ff = W.shape

    # 2) 提取 E=128 个 expert 的子权重 & 特征
    patterns = ffn_self.patterns.bool().cpu().numpy()  # (128, d_ff)
    E = patterns.shape[0]
    subWs = []
    feats = []
    for e in range(E):
        mask_e = patterns[e]
        We     = W[:, mask_e]             # (d_h, d_sub)
        subWs.append(We)
        feats.append(We.mean(dim=1).cpu().numpy())
    feats = np.stack(feats, axis=0)       # (128, d_h)

    # 3) KMeans 聚成 G 组
    feats_norm = normalize(feats, axis=1)
    km = KMeans(n_clusters=G, random_state=0).fit(feats_norm)
    labels = km.labels_                   # (128,)
    ffn_self.expert2group = labels

    # 4) 计算每组 prototype
    d_sub = subWs[0].shape[1]
    proto  = torch.zeros((G, d_h, d_sub), device=W.device)
    counts = torch.zeros(G, device=W.device)
    for e, g in enumerate(labels):
        proto[g]  += subWs[e]
        counts[g] += 1
    proto /= counts.view(G, 1, 1)         # (G, d_h, d_sub)
    ffn_self.prototypes = proto

    # 5) 计算每个 expert 的 LoRA 差量
    ffn_self.lora_A = {}
    ffn_self.lora_B = {}
    for e in range(E):
        g     = labels[e]
        Delta = subWs[e] - proto[g]       # (d_h, d_sub)
        U, S, Vt = torch.linalg.svd(Delta, full_matrices=False)
        U_r  = U[:, :r]                   # (d_h, r)
        S_r  = S[:r].sqrt()               # (r,)
        Vt_r = Vt[:r, :]                  # (r, d_sub)
        A_e  = U_r * S_r.unsqueeze(0)     # (d_h, r)
        B_e  = S_r.unsqueeze(1) * Vt_r    # (r, d_sub)
        ffn_self.lora_A[e] = A_e
        ffn_self.lora_B[e] = B_e


def _forward_with_protolora(ffn_self, hidden_states):
    """
    Prototype + LoRA 差量前向（含 detach 路由）：
      1) 用原始 128 expert 路由（detach hidden_states）得到 top-k
      2) 重构每个 expert 的子激活（prototype+LoRA）并 scatter-add
      3) ReLU → 掩码 → dropout → sparse wo
    """
    B, L, d_h = hidden_states.shape
    d_ff       = ffn_self.patterns.size(1)
    d_sub      = ffn_self.prototypes.size(2)

    # 1) 路由 & 掩码（detach + clone）
    flat      = hidden_states.clone().detach().view(-1, d_h)  # (B*L, d_h)
    flat_norm = flat / flat.norm(dim=-1, keepdim=True)
    scores    = ffn_self.mlp(flat_norm)                       # (B*L, 128)
    topk      = torch.topk(scores, k=k, dim=-1)[1]            # (B*L, k)
    mask      = torch.nn.functional.embedding(topk, ffn_self.patterns) \
                     .sum(dim=1)                              # (B*L, d_ff)
    mask      = mask.view(B, L, d_ff).bool()                  # (B, L, d_ff)

    # 2) 重构被选 expert 的子激活 H_e ∈ ℝ^{B*L×d_sub}
    topk_flat = topk.view(-1)                                 # (B*L*k,)
    g_flat    = [ffn_self.expert2group[e.item()] for e in topk_flat]
    proto_flat= ffn_self.prototypes[g_flat]                   # (B*L*k, d_h, d_sub)
    A_flat    = torch.stack([ffn_self.lora_A[e.item()] for e in topk_flat], dim=0)
    B_flat    = torch.stack([ffn_self.lora_B[e.item()] for e in topk_flat], dim=0)

    hs_flat = hidden_states.view(-1, d_h).unsqueeze(1).expand(-1, k, -1)
    hs_flat = hs_flat.reshape(-1, d_h)                        # (B*L*k, d_h)

    H_proto = (hs_flat.unsqueeze(1) @ proto_flat).squeeze(1)  # (B*L*k, d_sub)
    delta   = (hs_flat @ A_flat).bmm(B_flat)                  # (B*L*k, d_sub)

    # 3) scatter-add 回 full H ∈ ℝ^{B*L×d_ff}
    H_full = hidden_states.new_zeros((B*L, d_ff))
    for idx, e in enumerate(topk_flat):
        cols = ffn_self.patterns[e]                           # bool mask (d_ff,)
        H_full[idx//k, cols] += (H_proto[idx] + delta[idx])
    H = H_full.view(B, L, d_ff)

    # 4) Activation + sparse wo
    H = ffn_self.act(H)
    H = H.masked_fill(~mask, 0.0)
    H = ffn_self.dropout(H)
    return ffn_self.wo(H)


def change_forward(model, k=20):
    # Patch encoder
    for layer in model.encoder.block:
        ffn = layer.layer[1].DenseReluDense
        prepare_prototype_lora(ffn)
        ffn.forward = types.MethodType(_forward_with_protolora, ffn)
    # Patch decoder
    for layer in model.decoder.block:
        ffn = layer.layer[2].DenseReluDense
        prepare_prototype_lora(ffn)
        ffn.forward = types.MethodType(_forward_with_protolora, ffn)


# Apply patch
change_forward(model, k)

# Evaluation on SST-2
pred = []
for inst in sst2_dev:
    input_ids   = tokenizer("sst2 sentence: " + inst['sentence'],
                            return_tensors="pt").input_ids.cuda()
    dec_input   = tokenizer("<extra_id_0>",
                            return_tensors="pt").input_ids.cuda()[:, :1]
    out         = model(input_ids=input_ids, labels=dec_input)
    pred.append(int(out.logits[:,0,1465] > out.logits[:,0,2841]) == inst['label'])

print("Acc", sum(pred)/len(pred), 'k', k)
