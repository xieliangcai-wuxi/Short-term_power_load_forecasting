import torch
import torch.nn as nn

# ==========================================
# 1. Patch Embedding Layer (模拟 Patching 技术)
# ==========================================
# ==========================================
# 1. Patch Embedding Layer (修复 padding 警告版)
# ==========================================
class PatchEmbedding(nn.Module):
    """
    使用 1D 卷积Patching 操作。
    """
    def __init__(self, in_dim, embed_dim, patch_size=12): # patch_size=12 (偶数)
        super().__init__()
        self.patch_size = patch_size
        
        # --- 计算 Padding ---
        # 如果 kernel=12, 需要总 padding=11
        # 左边补 5, 右边补 6
        pad_total = patch_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        # 1. 常数填充层 (ConstantPad1d 接收参数为 (左, 右))
        self.pad = nn.ConstantPad1d((pad_left, pad_right), 0)
        
        # 2. 卷积层 (Padding设为0，因为我们已经手动补过了)
        self.proj = nn.Conv1d(
            in_channels=in_dim, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=1, 
            padding=0 # [修改] 不再使用 'same'
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [Batch, Seq, In_Dim] -> [Batch, In_Dim, Seq]
        x = x.transpose(1, 2)
        
        x = self.pad(x)
        x = self.proj(x)
        
        # 转回 [Batch, Seq, Embed_Dim]
        x = x.transpose(1, 2) 
        x = self.norm(x)
        return self.activation(x)

# ==========================================
# 2. Cross-Modal Attention Fusion (注意力融合)
# ==========================================
class CrossModalFusion(nn.Module):
    def __init__(self, dim=64, nhead=4, dropout=0.2):
        super().__init__()
        # 主模态(Load) 查询 辅助模态(Image+Text)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x_main, x_context):
        # x_main: Query (Load)
        # x_context: Key/Value (Image + Text)
        attn_out, _ = self.cross_attn(query=x_main, key=x_context, value=x_context)
        
        # 残差连接：保留 Load 的原始物理信息
        x = self.norm1(x_main + attn_out)
        
        x2 = self.ffn(x)
        x = self.norm2(x + x2)
        return x

# ==========================================
# 3. 共享特征提取器 (主类)
# ==========================================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, load_dim=1, img_channel=2, vocab_size=30522, fusion_dim=128):
        super().__init__()
        
        # --- A. 负荷分支 (Load Branch) -> 引入 Patching ---
        # 我们把简单的 Linear 换成了 PatchEmbedding
        # patch_size=12 代表模型每次看半天的趋势，而不仅仅是一个点
        self.load_patcher = PatchEmbedding(in_dim=load_dim, embed_dim=64, patch_size=12)
        
        # --- B. 图像分支 (Image Branch) ---
        self.img_cnn = nn.Sequential(
            nn.Conv2d(img_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), # Image 特征也映射到 64
            nn.GELU()
        )
        
        # --- C. 文本分支 (Text Branch) ---
        self.text_embed = nn.Embedding(vocab_size, 4) 
        self.text_fc = nn.Linear(32 * 4, 32) # Text 特征映射到 32
        
        # --- D. 融合准备 ---
        self.dim_main = 64  # Load (Patched)
        self.dim_context_raw = 64 + 32 # Image(64) + Text(32) = 96
        
        # 上下文对齐层: 把 96 维压缩到 64 维
        self.context_project = nn.Linear(self.dim_context_raw, self.dim_main)
        
        # 实例化融合模块
        self.fusion_module = CrossModalFusion(dim=self.dim_main, nhead=4, dropout=0.3)
        
        # 最终输出映射
        if self.dim_main != fusion_dim:
            self.final_project = nn.Linear(self.dim_main, fusion_dim)
        else:
            self.final_project = nn.Identity()

    def forward(self, x_load, x_img, x_text):
        """
        注意：这里只接收 3 个参数，没有 x_time
        """
        B, Seq, _ = x_load.shape
        
        # 1. Load (Main) -> 使用 Patching 提取特征
        # Output: [B, Seq, 64]
        f_load = self.load_patcher(x_load)
        
        # 2. Image
        x_img_flat = x_img.view(B * Seq, 2, x_img.size(-2), x_img.size(-1))
        f_img = self.img_cnn(x_img_flat).view(B, Seq, -1) # [B, Seq, 64]
        
        # 3. Text
        f_text = self.text_embed(x_text.long()).view(B, Seq, -1)
        f_text = self.text_fc(f_text)  # [B, Seq, 32]
        
        # --- 构建注意力组 ---
        
        # Query: 负荷是核心，它包含了我们要预测的趋势
        x_main = f_load 
        
        # Key/Value: 图像和文本是辅助信息
        x_context_raw = torch.cat([f_img, f_text], dim=-1) # [B, Seq, 96]
        x_context = self.context_project(x_context_raw)    # [B, Seq, 64]
        
        # --- 交叉注意力融合 ---
        
        fused = self.fusion_module(x_main, x_context)
        
        # --- 最终输出 ---
        out = self.final_project(fused)
        
        return out