# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.functional as F

class L2DistLoss(nn.Module):
    def __init__(self):
        super(L2DistLoss, self).__init__()

    def forward(self, feat1, feat2):
        loss = torch.mean(torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=1)))
        if torch.isnan(loss).any():
            loss = 0.0
        return loss


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothCELoss, self).__init__()

        self.smoothing = smoothing

    def forward(self, output, target):
        log_prob = F.log_softmax(output, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DistillKLLoss(nn.Module):
    def __init__(self, T):
        super(DistillKLLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        if y_t is None:
            return 0.0

        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.size(0)
        return loss
        
class AFRLoss(nn.Module):
    def __init__(self):
        super(AFRLoss, self).__init__()

    def forward(self,
        logits: torch.Tensor,          # (N*(K + β), num_classes)
        Hs: torch.Tensor,             # (N, K + β, d)
        P_bar: torch.Tensor,          # (N, β, d)
        support_features: torch.Tensor,  # (N*K, d)
        labels: torch.Tensor,         # (N*(K + β),)
        way_num: int,                 # N
        shot_num: int,                # K
        beta: int,                    # β
        mu1: float = 5.0,             # 权重 µ1
        mu2: float = 20.0,            # 权重 µ2
        tau: float = 0.05             # 温度系数 τ
    ):
        """
        根据论文的公式 (8)(9)(10)(11) 计算:
        
            L = L_CE + µ1 * L_SC + µ2 * L_MSE
        
        参数说明:
            logits:        AFRClassifier 的输出 logits,   形状 (N*(K + β), num_classes)
            Hs:            AFRClassifier 的输出 Hs,       形状 (N, K + β, d)
            P_bar:         AFRClassifier 的输出 P_bar,    形状 (N, β, d)
            support_features: 原始支持集特征,            形状 (N*K, d)
            labels:        每个特征的标签,               形状 (N*(K + β),)
                            其中同一个类的所有特征 label 相同 (比如 s)
            way_num:       N
            shot_num:      K
            beta:          每类校正原型个数
            mu1:           L_SC 的损失权重
            mu2:           L_MSE 的损失权重
            tau:           SC 损失中的温度系数
        
        返回:
            total_loss, (L_CE, L_SC, L_MSE)
        """
    
        # ------------------------------------------------------------------
        # 1) 交叉熵损失 L_CE
        # 在 PyTorch 中, nn.CrossEntropyLoss 默认对整个 batch 做平均
        # 因此无需显式再除以 (N*(K + β)) 。
        # 如果想与公式完全一致，可把 reduction='sum' 再手动 / (N*(K+β)) 也行。
        # 这里直接用默认的 mean 版就好。
        # ------------------------------------------------------------------
        ce_loss = F.cross_entropy(logits, labels)  # 标量
    
        # ------------------------------------------------------------------
        # 2) 监督式对比损失 L_SC
        # 这里演示一种典型的"SupCon"写法：
        #   - 先把 Hs 展平 => (N*(K+β), d)
        #   - 对每个 anchor, 它的 positive 就是同类、除自身以外的样本
        #   - 做 log(...) 这部分的运算
        # 也可逐类去循环，但要注意和论文公式对应。
        # ------------------------------------------------------------------
        Hs_flat = Hs.view(-1, Hs.size(-1))  # (M, d), 其中 M = N*(K+β)
        device = Hs_flat.device
    
        # 计算相似度矩阵: sim(i, j) = <Hs_flat[i], Hs_flat[j]>
        # 注意是否需要先行归一化(看你需求), 这里简单用点积
        sim_matrix = torch.matmul(Hs_flat, Hs_flat.T)  # (M, M)
        # 再除以温度系数 τ
        sim_matrix = sim_matrix / tau
    
        # labels: (M,) 其中同一类的所有 H_s 特征标签相同
        labels = labels.view(-1)
    
        # 构建各种 Mask
        M = way_num * (shot_num + beta)
        diag_mask  = torch.eye(M, dtype=torch.bool, device=device)    # 用于去掉对角线(自己 vs 自己)
        same_label = (labels.unsqueeze(1) == labels.unsqueeze(0))     # (M, M), 同类为 True
        pos_mask   = same_label & ~diag_mask                           # 同类且不是自己
        neg_mask   = ~same_label                                       # 不同类
    
        # 计算分母: 对每个样本 i，只在“不同类”的那些列上做 exp(sim)，再求和
        exp_sim = torch.exp(sim_matrix)               # (M, M)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)     # (M,)
    
        # 现在对分子：同类且非自己 => pos_mask
        # pos_mask[i] 表示行 i 的所有正样本列 j
        # 对于每个正样本 (i,j)，分子就是 exp(sim(i,j))，分母是 neg_sum[i]
        # 因此令
        #    SC_ij = log( exp(sim(i,j)) / neg_sum[i] )
        # 最后对所有正样本 (i,j) 再做平均即可
        i_idx, j_idx = pos_mask.nonzero(as_tuple=True)           # 所有 (i,j) 正样本对
        pos_exp_ij = exp_sim[i_idx, j_idx]                     # 与上面的 (i,j) 一一对应
        neg_sum_i = neg_sum[i_idx]                            # 每个对儿 (i_idx, j_idx) => row i_idx
    
        eps = 1e-12
        sc_ij = torch.log( (pos_exp_ij + eps) / (neg_sum_i + eps) )
        sc_loss = - sc_ij.mean()      # 因为公式里有一个 log(...)，故这里直接取负号相当于最小化
        # ------------------------------------------------------------------
        # 3) 均方误差 L_MSE
        #   L_MSE = 1/N ∑_{s=1 to N} || mean(f_s) - mean(p̄_s) ||^2
        # 其中:
        #   - mean(f_s) 即第 s 类所有支持样本的平均
        #   - mean(p̄_s) 即第 s 类 P_bar 的平均
        # 注意: support_features 排序: [s=0的K个, s=1的K个, ..., s=N-1的K个]
        # ------------------------------------------------------------------
        mse_loss = 0.0
        # support_features 形状: (N*K, d)
        support_features = support_features.view(way_num, shot_num, -1)  # => (N, K, d)
        for s in range(way_num):
            fs_mean = support_features[s].mean(dim=0)     # (d,)
            p_bar_mean = P_bar[s].mean(dim=0)             # (d,)
            mse_loss += F.mse_loss(fs_mean, p_bar_mean, reduction='sum')
        # 按论文 (9) 做 "1/N * ||...||^2"
        mse_loss = mse_loss / way_num
    
        # ------------------------------------------------------------------
        # 4) 加权求和
        # ------------------------------------------------------------------
        total_loss = ce_loss + mu1 * sc_loss + mu2 * mse_loss
    
        return total_loss, (ce_loss.item(), sc_loss.item(), mse_loss.item())