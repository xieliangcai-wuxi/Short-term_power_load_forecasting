import torch
import torch.nn as nn

class SharedFeatureExtractor(nn.Module):
    """
    Multimodal Feature Extractor for Power Forecasting.
    
    Architecture Design:
    1. Load Branch: MLP for numerical time-series.
    2. Image Branch: Resolution-agnostic CNN with Adaptive Pooling.
    3. Text Branch: Embedding + Linear projection.
    """
    def __init__(self, load_dim=1, img_channel=2, vocab_size=30522, embed_dim=32, fusion_dim=128):
        super().__init__()
        
        # --- A. 负荷分支 (Load Branch) ---
        self.load_mlp = nn.Sequential(
            nn.Linear(load_dim, 32),
            nn.GELU(), # Upgrade: ReLU -> GELU for smoother gradients
            nn.Linear(32, 32)
        )
        
        # --- B. 图像分支 (Image Branch - CNN) ---
        # 改进核心：移除硬编码的 Flatten 维度，使用 AdaptiveAvgPool2d
        self.img_cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(img_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2), # 64->32 (or 32->16)
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2), # 32->16 (or 16->8)
            
            # [顶刊通用性改进] 自适应池化层
            # 无论输入分辨率是 32x32 还是 64x64，强行压缩至 4x4
            # 输出维度固定为: [Batch, 32(Channel), 4(H), 4(W)]
            nn.AdaptiveAvgPool2d((4, 4)), 
            
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), # 32*16 = 512 -> 64
            nn.GELU()
        )
        
        # --- C. 文本分支 (Text Branch) ---
        self.text_embed = nn.Embedding(vocab_size, 4) 
        self.text_fc = nn.Linear(32 * 4, 32) 
        
        # --- D. 融合层 ---
        self.current_dim = 32 + 64 + 32 # 128
        
        if self.current_dim != fusion_dim:
            self.project = nn.Linear(self.current_dim, fusion_dim)
        else:
            self.project = nn.Identity()

    def forward(self, x_load, x_img, x_text):
        """
        Args:
            x_load: [Batch, Seq, 1]
            x_img:  [Batch, Seq, 2, H, W] (Supports variable H, W)
            x_text: [Batch, Seq, 32]
        Returns:
            Fused Features: [Batch, Seq, fusion_dim]
        """
        B, Seq, _ = x_load.shape
        
        # 1. Load Feature
        f_load = self.load_mlp(x_load) # [B, Seq, 32]
        
        # 2. Image Feature
        # Merge Batch & Seq dimensions: [B*Seq, C, H, W]
        x_img_flat = x_img.view(B * Seq, 2, x_img.size(-2), x_img.size(-1))
        f_img = self.img_cnn(x_img_flat)
        f_img = f_img.view(B, Seq, -1)   # [B, Seq, 64]
        
        # 3. Text Feature
        f_text = self.text_embed(x_text.long())
        f_text = f_text.view(B, Seq, -1) # Flatten tokens
        f_text = self.text_fc(f_text)    # [B, Seq, 32]
        
        # 4. Fusion
        f_cat = torch.cat([f_load, f_img, f_text], dim=-1) # [B, Seq, 128]
        out = self.project(f_cat)
        
        return out