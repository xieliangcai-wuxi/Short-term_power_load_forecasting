import torch
import torch.nn as nn
# 公共特征提取器 (Shared Feature Extractor)
class SharedFeatureExtractor(nn.Module):
    def __init__(self, load_dim=1, img_channel=2, vocab_size=30522, embed_dim=32, fusion_dim=128):
        super().__init__()
        
        # A. 负荷分支 (Load Branch)
        self.load_mlp = nn.Sequential(
            nn.Linear(load_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # B. 图像分支 (Image Branch - CNN)
        # 输入形状: [Batch*Seq, 2, 32, 32]
        self.img_cnn = nn.Sequential(
            nn.Conv2d(img_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32 -> 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16 -> 8x8
            
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU()
        )
        
        # C. 文本分支 (Text Branch - Embedding)
        self.text_embed = nn.Embedding(vocab_size, 4) 
        self.text_fc = nn.Linear(32 * 4, 32) 
        
        # 融合后的维度: 32 + 64 + 32 = 128
        self.current_dim = 32 + 64 + 32
        
        # 维度对齐层
        if self.current_dim != fusion_dim:
            self.project = nn.Linear(self.current_dim, fusion_dim)
        else:
            self.project = nn.Identity()

    def forward(self, x_load, x_img, x_text):
        B, Seq, _ = x_load.shape
        
        # 1. Load Feature
        f_load = self.load_mlp(x_load) # [B, Seq, 32]
        
        # 2. Image Feature (Merge Batch & Seq dimensions for CNN)
        x_img_flat = x_img.view(B * Seq, 2, 32, 32)
        f_img = self.img_cnn(x_img_flat)
        f_img = f_img.view(B, Seq, -1)   # [B, Seq, 64]
        
        # 3. Text Feature
        f_text = self.text_embed(x_text.long())
        f_text = f_text.view(B, Seq, -1)
        f_text = self.text_fc(f_text)    # [B, Seq, 32]
        
        # 4. Fusion
        f_cat = torch.cat([f_load, f_img, f_text], dim=-1) # [B, Seq, 128]
        out = self.project(f_cat)
        
        return out

