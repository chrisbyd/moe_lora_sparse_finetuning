from typing import DefaultDict
import sys
import torch
import os
import tqdm
from collections import Counter
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from k_means_constrained import KMeansConstrained



class ModelConfig:

    def __init__(self, filename, folder, split_num):
        self.filename = filename
        self.folder = folder
        self.split_num = split_num



class ParamSplit(LayerSplit):

    def __init__(self, config: ModelConfig, template, layer=0):
        super().__init__(config, template=template, layer=layer)
        self.type = 'param_split'

    def split(self):
        # 1) 加载原始 wi.weight，shape = (E, D)
        self.load_param()
        # 假设 D == d_hidden * d_ffn
        E, D = self.ffn_weight.shape
        d_h = self.config.hidden_size
        d_f = D // d_h

        # 2) 从原始扁平化权重恢复每个专家的 wi.weight 子矩阵
        W = torch.from_numpy(self.ffn_weight).view(E, d_h, d_f).cuda()

        # 3) 用 KMeans 先对 E 个专家做分组
        feats = normalize(self.ffn_weight, axis=1)            # (E, D)
        km = KMeans(n_clusters=self.config.split_num, random_state=0).fit(feats)
        labels = km.labels_                                   # (E,)
        self.expert2group = labels.tolist()

        # 4) 计算每组的 prototype —— 组内专家权重的简单平均
        G = self.config.split_num
        proto = torch.zeros((G, d_h, d_f), device=W.device)
        counts = torch.zeros(G, device=W.device)
        for e, g in enumerate(labels):
            proto[g] += W[e]
            counts[g] += 1
        proto /= counts.view(G, 1, 1)                         # (G, d_h, d_f)
        self.prototypes = proto

        # 5) 计算每个专家的 ΔW 并做 LoRA 分解
        r = self.config.lora_rank
        self.lora_A = {}
        self.lora_B = {}
        for e in range(E):
            g = labels[e]
            Delta = W[e] - proto[g]                           # (d_h, d_f)
            U, S, Vt = torch.linalg.svd(Delta, full_matrices=False)
            # 截断到秩 r
            U_r = U[:, :r]                                    # (d_h, r)
            S_r = S[:r].sqrt()                                # (r,)
            Vt_r = Vt[:r, :]                                  # (r, d_f)
            # 构造 A_e, B_e
            A_e = U_r * S_r.unsqueeze(0)                      # (d_h, r)
            B_e = S_r.unsqueeze(1) * Vt_r                     # (r, d_f)
            self.lora_A[e] = A_e
            self.lora_B[e] = B_e

        # 6) 最后把 prototype + LoRA 注入到对应的 FFN 模块里
        self.inject_to_ffn()

    def inject_to_ffn(self):
        for e, ffn in enumerate(self.find_ffn_modules()):
            g = self.expert2group[e]
            # 注入这层、这组的 prototype 和 专家差量
            ffn.proto_W = self.prototypes[g]
            ffn.lora_A = self.lora_A[e]
            ffn.lora_B = self.lora_B[e]
            # 用前面给出的 _forward_with_lora 取代原 forward
            ffn.forward = types.MethodType(_forward_with_lora, ffn)






class LayerSplit:

    def __init__(self, config : ModelConfig, template, layer=0):
        self.config = config
        self.layer = layer
        self.template = template

    def split(self):
        pass
    
    def save(self):
        save_folder = os.path.join(self.config.folder, self.type)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        filename = os.path.join(save_folder, self.template.format(self.layer))
        torch.save(self.labels, filename)

    def cnt(self):
        print(Counter(self.labels))

    def load_param(self):
        self.ffn_weight = load_ffn_weight(self.config.filename, self.template, self.layer)
        self.neuron_num = self.ffn_weight.shape[0]
        self.split_size = self.neuron_num // self.config.split_num
        assert self.split_size * self.config.split_num == self.neuron_num

def load_ffn_weight(filename, template, layer):

    model = torch.load(filename, map_location='cpu')
    key = template.format(layer)

    return model[key].numpy()