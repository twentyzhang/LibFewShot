import torch
import torch.nn as nn
from core.utils import accuracy
from .finetuning_model import FinetuningModel
import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from core.model.loss import AFRLoss
import torch.nn.functional as F

class PrototypeDataAugmentor:
    def __init__(self, emb_func, train_csv, avg_feature_file, device="cuda"):
        """
        初始化增强器
        :param emb_func: 特征提取器（此处不再使用）
        :param train_csv: 包含类别标签的 CSV 文件路径
        :param avg_feature_file: 类别与其平均特征的映射表文件路径
        :param device: 计算设备
        """
        self.device = device
        self.train_csv = os.path.join(os.path.dirname(avg_feature_file), 'id2cate.csv')
        self.avg_feature_file = os.path.join(os.path.dirname(avg_feature_file), 'category_features.csv')

        # 加载平均特征表
        self.avg_feature_df = pd.read_csv(self.avg_feature_file, header=0)
        self.avg_feature_dict = {
            row.iloc[0]: torch.tensor(row.iloc[1:].values.astype(np.float32), dtype=torch.float32).to(self.device)
            for _, row in self.avg_feature_df.iterrows()
        }

    def find_closest_classes_for_episode(self, closest_words_file, episode_target, top_k=3, train=True):
        """
        为一个 episode 中的所有类别找到最接近的类别。
        :param closest_words_file: 包含相似类别的文件路径
        :param episode_target: 当前 episode 的类别列表
        :param top_k: 每个类别获取的最接近类别数
        :return: 每个类别的最接近类别列表
        """
        if train:
            closest_df = pd.read_csv(closest_words_file)
            closest_dict = {row['id']: row.values[1:].tolist() for _, row in closest_df.iterrows()}
    
            class_id = episode_target
            closest_classes = closest_dict.get(class_id, [])
            return closest_classes[:top_k]
    
            return closest_classes_per_episode
        else:
            closest_df = pd.read_csv(closest_words_file)
            closest_dict = {row['id']: row.values[1:].tolist() for _, row in closest_df.iterrows()}
    
            closest_classes_per_episode = []
            for class_id in episode_target:
                closest_classes = closest_dict.get(class_id, [])
                excluded_classes = set(episode_target) - {class_id}
                closest_classes = [c for c in closest_classes if c not in excluded_classes]
                closest_classes_per_episode.append(closest_classes[:top_k])
    
            return closest_classes_per_episode

    def load_class_samples(self, class_id, closest_classes, class_label_dict):
        """
        加载与给定类别最接近的类别的平均特征。
        :param class_id: 当前任务类别 ID
        :param closest_classes: 与当前类最接近的类别列表
        :return: 增强的特征和对应的类别标签
        """
        all_features = []
        all_targets = []
        for closest_class in closest_classes:
            if closest_class in self.avg_feature_dict:
                all_features.append(self.avg_feature_dict[closest_class])
                all_targets.append(class_id)

        if not all_features:
            raise ValueError(f"No features found for class {class_id} and its closest classes.")
            
        numeric_targets = [class_label_dict[target] for target in all_targets]
        return torch.stack(all_features), torch.tensor(numeric_targets, device=self.device)

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

    def forward(self, support_features: torch.Tensor, P: torch.Tensor):
        """
        参数:
            support_features: (way_num * shot_num, d)
            P:                (way_num, beta_s, d)

        返回:
            logits: shape (way_num * (shot_num + beta_s), num_classes),
            Hs:     shape (way_num, shot_num + beta_s, d),
            P_bar:  shape (way_num, beta_s, d)
        """
        # 1) 调用 AFR 得到校正后的原型 P_bar 和组合特征 Hs
        P_bar, Hs = self.afr(support_features, P)

        # 2) 展平 Hs 送入线性层得到 logits
        Hs_flat = Hs.view(-1, Hs.size(-1))  # => (way_num*(shot_num + beta_s), d)
        logits = self.fc(Hs_flat)          # => (way_num*(shot_num + beta_s), num_classes)

        return logits, P_bar, Hs      
        
class AFR(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(AFR, self).__init__(**kwargs)
        self.AFRmodule = AFRClassifier(d=feat_dim, num_classes = num_class, )
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param
        self.augmentor = PrototypeDataAugmentor(
            emb_func=None,  # 不再需要特征提取器
            train_csv=kwargs['data_root'],
            avg_feature_file=kwargs['data_root'],
        )
        self.closest_words_file = os.path.join(kwargs['data_root'], 'closest_words.csv')
        self.class_label_dict = kwargs["class_label_dict"]
        
    
    def set_forward(self, batch):
        """
        前向计算，包括特征增强。
        """        
        # 特征提取
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        class_label_dict = {**self.class_label_dict[0].dataset._generate_data_list()[0],
                            **self.class_label_dict[0].dataset._generate_data_list()[1]}  # label -> class
        # 创建逆映射字典
        inverse_class_label_dict = {v: k for k, v in class_label_dict.items()} # class -> label(nxxxx)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        current_classes = support_feat.tolist()
        # 使用逆映射字典来将 augmented_target 的值映射回原始的键
        reversed_target = [inverse_class_label_dict.get(element, element) for element in current_classes]
        support_target = support_target.to(self.device)
        episode_size = feat.size(0)

        # 找到每行类别的最接近类别
        closest_classes_per_episode = [
            self.augmentor.find_closest_classes_for_episode(
                self.closest_words_file, episode_target=classes, train=True
            ) for classes in reversed_target
        ]
        mapped_closest_target = [[class_label_dict.get(element, element) for element in sublist] for sublist in closest_classes_per_episode]
        # 使用已提供的平均特征进行增强
        augmented_feat = []
        augmented_target = []
        for i, closest_classes in enumerate(closest_classes_per_episode): 
            class_id = reversed_target[i]
            closest_class_list = closest_classes
            aug_feat, aug_target = self.augmentor.load_class_samples(
                class_id, closest_class_list, class_label_dict
            )# closest_classes为纯数字
            augmented_feat.append(aug_feat)
            augmented_target.append(aug_target)
        
        augmented_feat = torch.cat(augmented_feat, dim=0).view(episode_size, 3, -1)
        augmented_target = torch.cat(augmented_target, dim=0).view(episode_size, 3, -1)



        # 拼接增强特征与支持集特征
        combined_support_feat = torch.cat([support_feat, augmented_feat], dim=0)
        combined_support_target = torch.cat([support_target, augmented_target.view(-1)], dim=0)
        
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(
                combined_support_feat[i], combined_support_target[i], query_feat[i]
            )
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        
        logits, P_bar, Hs = self.AFRmodule(feat, augmented_feat)

        self.loss_func = AFRLoss()
        loss,_ = self.loss_func(logits, 
                       Hs, 
                       P_bar, 
                       feat, 
                       combined_support_target, 
                      episode_size,
                      1,
                      3)

        output = logits[:episode_size]
        acc = accuracy(output, global_target.reshape(-1))


        return output, acc

    def set_forward_loss(self, batch):
        """
        前向计算，包括特征增强。
        """
        class_label_dict = self.class_label_dict[0].dataset._generate_data_list()[2] # label -> class
        # 创建逆映射字典
        inverse_class_label_dict = {v: k for k, v in class_label_dict.items()} # class -> label(nxxxx)

        image, global_target = batch
        current_classes = global_target.tolist()
        # 使用逆映射字典来将 augmented_target 的值映射回原始的键
        reversed_target = [inverse_class_label_dict.get(element, element) for element in current_classes]

        global_target = global_target.to(self.device)
        image = image.to(self.device)
        # 特征提取
        with torch.no_grad():
            feat = self.emb_func(image)
        episode_size = feat.size(0)
        
        # 找到每行类别的最接近类别
        closest_classes_per_episode = [
            self.augmentor.find_closest_classes_for_episode(
                self.closest_words_file, episode_target=classes, train=True
            ) for classes in reversed_target
        ]
        mapped_closest_target = [[class_label_dict.get(element, element) for element in sublist] for sublist in closest_classes_per_episode]
        # 使用已提供的平均特征进行增强
        augmented_feat = []
        augmented_target = []
        for i, closest_classes in enumerate(closest_classes_per_episode): 
            class_id = reversed_target[i]
            closest_class_list = closest_classes
            aug_feat, aug_target = self.augmentor.load_class_samples(
                class_id, closest_class_list, class_label_dict
            )# closest_classes为纯数字
            augmented_feat.append(aug_feat)
            augmented_target.append(aug_target)
        
        augmented_feat = torch.cat(augmented_feat, dim=0).view(episode_size, 3, -1)
        augmented_target = torch.cat(augmented_target, dim=0).view(episode_size, 3, -1)



        # 拼接增强特征与支持集特征
        # combined_support_feat = torch.cat([support_feat, augmented_feat], dim=0)
        combined_support_target = torch.cat([global_target, augmented_target.view(-1)], dim=0)
        logits, P_bar, Hs = self.AFRmodule(feat, augmented_feat)

        self.loss_func = AFRLoss()
        loss,_ = self.loss_func(logits, 
                       Hs, 
                       P_bar, 
                       feat, 
                       combined_support_target, 
                      episode_size,
                      1,
                      3)

        output = logits[:episode_size]
        acc = accuracy(output, global_target.reshape(-1))

        return output, acc, loss

    
    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = nn.Linear(self.feat_dim, self.num_class)
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)
        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                select_id = rand_id[
                    i : min(i + self.inner_param["inner_batch_size"], support_size)
                ]
                batch = support_feat[select_id]
                target = support_target[select_id]

                output = classifier(batch)

                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output = classifier(query_feat)
        return output