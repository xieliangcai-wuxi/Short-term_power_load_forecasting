import torch
import torch.nn as nn
# 确保引用路径正确，如果在 model 文件夹内，用 .SharedFeatureExtractor
from .SharedFeatureExtractor import SharedFeatureExtractor

class MMTransformerModel(nn.Module):
    def __init__(self, seq_len=168, pred_len=24, d_model=128, n_layers=2, nhead=4):
        super().__init__()
        # 1. 公共特征提取器 (保持一致)
        self.feature_extractor = SharedFeatureExtractor(fusion_dim=d_model)
        
        # 2. 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 3. 归一化层 (严谨变量控制：Mamba 和 LSTM 都有这层)
        self.norm = nn.LayerNorm(d_model)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        
        # *** 关键修正 ***：将 norm 传给 TransformerEncoder
        # 这样它会在所有层跑完后，自动执行一次 LayerNorm，这就和 Mamba 的逻辑对齐了
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=self.norm)
        
        # 5. 输出层
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x_load, x_img, x_text):
        # 特征提取
        x = self.feature_extractor(x_load, x_img, x_text)
        
        # 加上位置编码
        x = x + self.pos_encoder
        
        # Transformer 计算 (内部最后会自动调用 self.norm)
        x = self.transformer(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]
        
        # 预测并调整维度 [B, 24, 1]
        return self.fc_out(x).unsqueeze(-1)