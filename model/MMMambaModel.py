import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel
from .SharedFeatureExtractor import SharedFeatureExtractor

class MMMambaModel(nn.Module):
    def __init__(self, seq_len=168, pred_len=24, d_model=128, n_layers=2):
        super().__init__()
        
        
        self.feature_extractor = SharedFeatureExtractor(fusion_dim=d_model)
        
        # 2. 配置 Hugging Face Mamba
        # 注意：我们需要将你的参数映射到 HF 的配置参数名
        config = MambaConfig(
            hidden_size=d_model,          # 对应 d_model
            num_hidden_layers=n_layers,   # 对应 n_layers
            state_size=16,                # 默认 SSM 状态维度 (d_state)
            expand=2,                     # 扩展因子
            conv_kernel=4,                # 卷积核大小 (d_conv)
            use_bias=True,                # 是否使用 bias
            use_cache=False               # 训练时设为 False
        )
        
        # 3. 初始化官方 Mamba 模型
        # 这是一个裸模型，不包含 Embedding 层，非常适合直接接时序特征
        self.mamba = MambaModel(config)
        
        # 4. 输出层
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x_load, x_img, x_text):
        # [Step 1] 特征提取
        # x shape: [Batch_Size, Seq_Len, d_model]
        x = self.feature_extractor(x_load, x_img, x_text)
        
        # [Step 2] 输入 Mamba
        outputs = self.mamba(inputs_embeds=x)
        
        # outputs.last_hidden_state shape: [Batch, Seq, d_model]
        hidden_states = outputs.last_hidden_state
        
        # [Step 3] 取最后一个时间步进行预测
        # 与 LSTM/Transformer 保持一致，取序列最后一个点的输出
        last_step_feature = hidden_states[:, -1, :] 
        
        # LayerNorm (HF Mamba 内部通常已有 Norm，但加一个保证稳定性)
        last_step_feature = self.norm(last_step_feature)
        
        # [Step 4] 线性映射到预测长度
        out = self.fc_out(last_step_feature)
        
        # [Step 5] 调整维度以匹配 Label: [Batch, Pred_Len, 1]
        return out.unsqueeze(-1)