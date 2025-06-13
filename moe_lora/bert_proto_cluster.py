import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import T5Config
class ParamSplit(LayerSplit):
    def __init__(self, config: ModelConfig, template, layer=0):
        super().__init__(config, template=template, layer=layer)
        self.type = 'param_split'

    def split(self):
        # 1. 加载并归一化专家权重特征
        self.load_param()  # 假设 self.ffn_weight: np.ndarray, shape=(E, D)
        feats = normalize(self.ffn_weight, axis=1)  # (E, D)

        # 2. 聚类为 G 个组
        G = self.config.split_num
        kmeans = KMeans(n_clusters=G, random_state=0).fit(feats)
        labels = kmeans.labels_  # (E,)
        self.expert2group = labels.tolist()

        # 3. 重塑原始权重为矩阵形状
        E, D = self.ffn_weight.shape
        d_hidden = self.config.hidden_size
        d_ffn = D // d_hidden
        W = torch.from_numpy(self.ffn_weight).view(E, d_hidden, d_ffn)

        # 4. 计算各组原型（prototype）
        proto_W = torch.zeros((G, d_hidden, d_ffn), device=W.device)
        counts = torch.zeros(G, device=W.device)
        for e, g in enumerate(labels):
            proto_W[g] += W[e]
            counts[g] += 1
        proto_W = proto_W / counts.view(G, 1, 1)
        self.prototypes = proto_W  # shape=(G, d_hidden, d_ffn)

        # 5. 计算 LoRA 差量并分解为 A,B
        r = self.config.lora_rank
        self.lora_A = {}
        self.lora_B = {}
        for e in range(E):
            g = labels[e]
            Delta = W[e] - proto_W[g]  # (d_hidden, d_ffn)
            # SVD 分解
            U, S, Vt = torch.linalg.svd(Delta, full_matrices=False)
            U_r = U[:, :r]
            S_r = S[:r].sqrt()
            Vt_r = Vt[:r, :]
            # A_e = U_r * S_r, B_e = S_r * Vt_r
            A_e = U_r * S_r.unsqueeze(0)
            B_e = S_r.unsqueeze(1) * Vt_r
            self.lora_A[e] = A_e
            self.lora_B[e] = B_e

        # 6. 将 prototype 和 LoRA 注入到 ffn 模块
        self.inject_to_ffn()

    def inject_to_ffn(self):
        """
        将计算好的 prototypes, lora_A, lora_B 注入到对应的 T5DenseActDense 模块
        """
        for e, ffn in enumerate(self.find_ffn_modules()):
            g = self.expert2group[e]
            ffn.proto_W = self.prototypes[g]
            ffn.lora_A = self.lora_A[e]
            ffn.lora_B = self.lora_B[e]
            # 替换 forward
            ffn.forward = types.MethodType(_forward_with_lora, ffn)

# _forward_with_lora 请参照前面示例，将 ffn.proto_W, ffn.lora_A/B 用于前向计算
