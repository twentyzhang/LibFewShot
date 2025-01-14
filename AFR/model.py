import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# 1. Instance Attention（原型生成 + Self-Attention 校正）
# ============================================
class InstanceAttention(nn.Module):
    """
    实例注意力模块 (参照论文中 Figure 3 对原型进行分布校正)
    公式参考：
        Q = f_s * W_q
        K = P * W_k
        V = P * W_v
        A_s = softmax(Q * K^T / sqrt(d)) * V
        P_hat = ReLU(A_s * W_p) + P

    仅支持批量输入:
        - f_s: (way_num, d)  
        - P:   (way_num, beta_s, d)
    """
    def __init__(self, d: int):
        """
        参数:
            d: 特征维度 (与 CNN 输出特征同维度)
        """
        super().__init__()
        self.d = d

        # 学习参数
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_p = nn.Linear(d, d, bias=False)

        # 激活
        self.relu = nn.ReLU()

    def forward(self, f_s: torch.Tensor, P: torch.Tensor):
        """
        参数:
            f_s: shape (way_num, d)
            P:   shape (way_num, beta_s, d)

        返回:
            P_hat: shape (way_num, beta_s, d)
        """
        # ========== 形状检查 ==========
        assert f_s.dim() == 2, "f_s 必须是 (way_num, d)"
        assert P.dim() == 3,   "P 必须是 (way_num, beta_s, d)"
        way_num, d_ = f_s.shape
        way_num2, beta_s, d2 = P.shape
        assert way_num == way_num2,  "f_s 与 P 的 way_num 不匹配"
        assert d_ == d2, "f_s 与 P 的特征维度不匹配"

        # =========== 1) 计算 Q, K, V ===========
        # f_s: (way_num , d)
        # P:   (way_num, beta_s, d)
        Q = self.W_q(f_s)       # => (way_num, d)
        K = self.W_k(P)         # => (way_num, beta_s, d)
        V = self.W_v(P)         # => (way_num, beta_s, d)

        # =========== 2) 计算注意力权重 attn_weights ===========
        #   Q: (way_num, d) => 视作 (way_num, 1, d)
        #   K: (way_num, beta_s, d) => K^T: (way_num, d, beta_s)
        # => attn_scores: (way_num, beta_s)
        Q_ = Q.unsqueeze(1)                      # => (way_num, 1, d)
        K_t = K.transpose(1, 2)                  # => (way_num, d, beta_s)
        attn_scores = torch.bmm(Q_, K_t) / (self.d ** 0.5)  # => (way_num, 1, beta_s)
        attn_scores = attn_scores.squeeze(1)     # => (way_num, beta_s)
        attn_weights = F.softmax(attn_scores, dim=-1)  # => (way_num, beta_s)

        # =========== 3) 计算 A_s 并校正 ===========
        # A_s = attn_weights * V => (way_num, d)
        attn_weights_ = attn_weights.unsqueeze(1)  # => (way_num, 1, beta_s)
        A_s = torch.bmm(attn_weights_, V)          # => (way_num, 1, d)
        A_s = A_s.squeeze(1)                       # => (way_num, d)

        # A_s_calibrated = ReLU(W_p(A_s)) => (way_num, d)
        A_s_calibrated = self.relu(self.W_p(A_s))

        # =========== 4) P_hat = P + A_s_calibrated (广播加和) ===========
        # (way_num, d) => unsqueeze(1) => (way_num, 1, d) => expand => (way_num, beta_s, d)
        A_s_expand = A_s_calibrated.unsqueeze(1).expand(way_num, beta_s, d2)
        P_hat = P + A_s_expand  # => (way_num, beta_s, d)

        return P_hat


# ============================================
# 2. Channel Attention（SE-Block）
# ============================================

class ChannelAttention(nn.Module):
    """
    通道注意力模块 (SE block)
    参考公式:
        E_s = sigma(FC2(ReLU(FC1(P_hat))))
        P_bar = E_s ⊙ P_hat + P

    仅支持:
        P_hat: (way_num, beta_s, d)
        P:     (way_num, beta_s, d)
    """
    def __init__(self, d: int, reduction: int = 16):
        """
        参数:
            d: 特征维度
            reduction: FC1 的降维倍数
        """
        super().__init__()
        self.d = d
        mid = max(d // reduction, 1)

        self.fc1 = nn.Linear(d, mid, bias=True)
        self.fc2 = nn.Linear(mid, d, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, P_hat: torch.Tensor, P: torch.Tensor):
        """
        参数:
            P_hat: shape (way_num, beta_s, d)
            P:     shape (way_num, beta_s, d)

        返回:
            P_bar: shape (way_num, beta_s, d)
        """
        # ====== 形状检查 ======
        assert P_hat.dim() == 3, "P_hat 必须是 (way_num, beta_s, d)"
        assert P.dim() == 3,     "P 必须是 (way_num, beta_s, d)"
        way_num, beta_s, d_ = P_hat.shape
        way_num2, beta_s2, d2 = P.shape
        assert way_num == way_num2, "P_hat 与 P 的 way_num 不匹配"
        assert beta_s == beta_s2,   "P_hat 与 P 的 beta_s 不匹配"
        assert d_ == d2,            "P_hat 与 P 的特征维度不匹配"

        # 1) 直接对 (way_num, beta_s, d) 做线性映射
        # fc1 -> ReLU -> fc2 -> Sigmoid
        # PyTorch 的 nn.Linear 可以沿着前面维度批量处理，只要最后一维 = d
        x = self.fc1(P_hat)     # => (way_num, beta_s, mid)
        x = self.relu(x)        # => (way_num, beta_s, mid)
        x = self.fc2(x)         # => (way_num, beta_s, d)
        x = self.sigmoid(x)     # => (way_num, beta_s, d)

        # 2) 残差融合: P_bar = x ⊙ P_hat + P
        P_bar = x * P_hat + P   # => (way_num, beta_s, d)

        return P_bar


# ============================================
# 3. 整合 AFR 主体模块
# ============================================
class AttentiveFeatureRegularization(nn.Module):
    """
    输入:
        support_features: (way_num * shot_num, d) —— 已排好类别顺序, 相邻 shot_num 个同类
        P:  (way_num, beta_s, d) —— 与之对应的原型集 (同 way_num)
    返回:
        P_bar: (way_num, beta_s, d)   —— 最终校正后的原型
    """
    def __init__(self, d: int, reduction: int = 16):
        """
        参数:
            d: 特征维度
            reduction: SE-Block 中的通道降维倍数
        """
        super().__init__()
        self.d = d
        self.instance_attention = InstanceAttention(d)
        self.channel_attention = ChannelAttention(d, reduction)

    def forward(self, support_features: torch.Tensor, P: torch.Tensor):
        """
        参数:
            support_features: shape (way_num * shot_num, d)
            P:                shape (way_num, beta_s, d)
        返回:
            P_bar: (way_num, beta_s, d)
        """
        assert support_features.dim() == 2, (
            "support_features 必须是 (way_num * shot_num, d)"
        )
        assert P.dim() == 3, "P 必须是 (way_num, beta_s, d)"

        way_num = P.size(0)
        # shot_num 可以自动计算
        support_size = support_features.size(0)
        assert support_size % way_num == 0, (
            f"support_features 总数 {support_size} 应能被 way_num {way_num} 整除!"
        )
        shot_num = support_size // way_num

        # 1) reshape + 求平均
        support_features = support_features.view(way_num, shot_num, self.d)
        f_s = support_features.mean(dim=1)  # => (way_num, d)

        # 2) Instance Attention
        P_hat = self.instance_attention(f_s, P)   # => (way_num, beta_s, d)

        # 3) Channel Attention
        P_bar = self.channel_attention(P_hat, P)  # => (way_num, beta_s, d)

        # 4) 组装 Hs: [f_s^(1), ..., f_s^(K), p¯_1^s, ..., p¯_βs^s]
        #    其中 support_features 的形状是 (way_num, shot_num, d)
        #    P_bar 的形状是 (way_num, beta_s, d)
        Hs = torch.cat([support_features, P_bar], dim=1)
        # Hs 形状: (way_num, shot_num + beta_s, d)

        return P_bar, Hs

# ==========================================
#    下面是关键：加一个"最后一层"线性层做分类
# ==========================================
class AFRClassifier(nn.Module):
    """
    使用 AFR + 最后一层线性层输出分类结果
    """
    def __init__(self, d: int, num_classes: int, reduction: int = 16):
        super().__init__()
        # 复用你实现的 AFR
        self.afr = AttentiveFeatureRegularization(d, reduction)
        # 最终线性层: (d -> num_classes)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, support_features: torch.Tensor, P: torch.Tensor, train = True):
        """
        参数:
            support_features: (way_num * shot_num, d)
            P:                (way_num, beta_s, d)

        返回:
            logits: shape (way_num * (shot_num + beta_s), num_classes),
            Hs:     shape (way_num, shot_num + beta_s, d),
            P_bar:  shape (way_num, beta_s, d)
        """
        if train == True:
            # 1) 调用 AFR 得到校正后的原型 P_bar 和组合特征 Hs
            P_bar, Hs = self.afr(support_features, P)

            # 2) 展平 Hs 送入线性层得到 logits
            Hs_flat = Hs.view(-1, Hs.size(-1))  # => (way_num*(shot_num + beta_s), d)
            logits = self.fc(Hs_flat)          # => (way_num*(shot_num + beta_s), num_classes)

            return logits, P_bar, Hs 
        else:
            logits = self.fc(support_features)


            return logits
           



if __name__ == "__main__":
    def test_afr_classifier():
        # 1) 准备超参数
        way_num = 5       # 类别数
        shot_num = 5      # 每类 support 样本数
        beta_s = 3        # 每类原型个数 (P 的第二维)
        d = 64            # 特征维度
        num_classes = 5   # 最终分类器输出维度，通常可与 way_num 相同
        
        # 2) 初始化模型
        model = AFRClassifier(d, num_classes)
        
        # 3) 构造随机输入:
        #    support_features: (way_num * shot_num, d)
        #    P: (way_num, beta_s, d)
        support_features = torch.randn(way_num * shot_num, d)
        P = torch.randn(way_num, beta_s, d)

        # 4) 前向计算
        logits, P_bar, Hs = model(support_features, P)

        # 5) 打印输出形状
        print(f"logits.shape: {logits.shape}")
        print(f"P_bar.shape:  {P_bar.shape}")
        print(f"Hs.shape:     {Hs.shape}")

        # 验证形状是否符合预期:
        # - logits: (way_num*(shot_num + beta_s), num_classes) => (5*(5+3), 5) => (40, 5)
        # - P_bar:  (way_num, beta_s, d) => (5, 3, 64)
        # - Hs:     (way_num, shot_num + beta_s, d) => (5, 8, 64)

    test_afr_classifier()
    

