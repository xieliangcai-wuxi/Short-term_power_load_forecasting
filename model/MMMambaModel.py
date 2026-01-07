import torch
import torch.nn as nn
from .SharedFeatureExtractor import SharedFeatureExtractor
from mamba_ssm import Mamba

class MMMambaModel(nn.Module):
    def __init__(self, seq_len=168, pred_len=24, d_model=128, n_layers=2):
        super().__init__()
        
        # 使用公共特征提取器
        self.feature_extractor = SharedFeatureExtractor(fusion_dim=d_model)
        
        # 调用官方 Mamba 模块
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x_load, x_img, x_text):
        # 1. 提取特征 [B, Seq, d_model]
        x = self.feature_extractor(x_load, x_img, x_text)
        
        # 2. Mamba 序列建模
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        # 3. 预测 (取最后一个时间步)
        x = x[:, -1, :] 
        out = self.fc_out(x)
        return out.unsqueeze(-1)