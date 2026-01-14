import torch
import torch.nn as nn
# 确保 SharedFeatureExtractor 已经是你修改过的 Patching+Attention 版本
try:
    from .SharedFeatureExtractor import SharedFeatureExtractor
except ImportError:
    from SharedFeatureExtractor import SharedFeatureExtractor

class MMGRUModel(nn.Module):
    def __init__(self, seq_len=168, pred_len=24, d_model=64, n_layers=2):
        """
        Args:
            d_model: 64 (与 FeatureExtractor 输出对齐)
            n_layers: 2 (双层 GRU 足够捕捉复杂模式)
        """
        super().__init__()
        self.pred_len = pred_len
        
        # 1. 强力特征提取器 (Patching + Attention Fusion)
        # 输出形状: [Batch, Seq, d_model]
        self.feature_extractor = SharedFeatureExtractor(fusion_dim=d_model)
        
        # 2. 双向 GRU 骨架 (Bi-Directional GRU)
        # input_size=64, hidden_size=64
        # bidirectional=True -> 输出维度会自动变成 64*2 = 128
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3, # 适当的 Dropout 防止过拟合
            bidirectional=True 
        )
        
        # 3. 输出层
        # 因为是双向，所以输入维度是 d_model * 2
        self.norm = nn.LayerNorm(d_model * 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3) # 在全连接层前再加一道防线
        
        # 映射回预测长度 [128 -> 24]
        self.fc_out = nn.Linear(d_model * 2, pred_len)

    def forward(self, x_load, x_img, x_text):
        # 1. 多模态融合
        # x shape: [Batch, 168, 64]
        x = self.feature_extractor(x_load, x_img, x_text)
        
        # 2. GRU 序列建模
        # out shape: [Batch, 168, 128] (64*2)
        out, _ = self.gru(x)
        
        # 3. 取最后一个时间步的信息 (Last Token)
        # 代表了“现在”的状态，包含了过去168小时的记忆
        out = out[:, -1, :] # [Batch, 128]
        
        # 4. 预测头
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 5. 生成预测结果
        # [Batch, 24] -> [Batch, 24, 1] 以匹配 Loss 格式
        prediction = self.fc_out(out).view(-1, self.pred_len, 1)
        
        return prediction