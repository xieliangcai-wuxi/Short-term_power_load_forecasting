import torch
import torch.nn as nn
from .SharedFeatureExtractor import SharedFeatureExtractor

class MMLSTMModel(nn.Module):
    def __init__(self, seq_len=168, pred_len=24, d_model=128, n_layers=2):
        super().__init__()
        self.pred_len = pred_len
        
        # 核心特征提取器：将 [B, 168, ...] 映射为 [B, 168, d_model]
        self.feature_extractor = SharedFeatureExtractor(fusion_dim=d_model)
        
        # LSTM：处理多模态融合后的序列特征
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=0.1 if n_layers > 1 else 0 # 只有多层时 dropout 才有效
        )
        
        # 顶刊标准：在预测头前加入规范化层，增强数值稳定性
        self.norm = nn.LayerNorm(d_model)
        
        # 预测输出层：将隐藏状态映射至未来长度
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x_load, x_img, x_text):
        """
        x_load: [B, 168, 1]
        x_img:  [B, 168, 2, 32, 32]
        x_text: [B, 168, 32]
        """
        # 1. 多模态特征融合
        # 返回形状: [Batch, 168, d_model]
        x = self.feature_extractor(x_load, x_img, x_text)
        
        # 2. 时序建模
        # out: [Batch, 168, d_model]
        out, _ = self.lstm(x)
        
        # 3. 提取末端隐藏状态 (Many-to-One 策略)
        # 取序列的最后一个时间步，因为它融合了之前 168 小时的所有上下文
        out = out[:, -1, :] 
        
        # 4. 规范化与预测
        out = self.norm(out)
        
        # 5. 映射到预测长度并重塑维度
        # 输出: [Batch, 24, 1]
        prediction = self.fc_out(out).view(-1, self.pred_len, 1)
        
        return prediction