import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm 

class MMTimeSeriesDataset(Dataset):
    def __init__(self, data_path, seq_len=168, pred_len=24, mode='train'):
        # 1. 读取 .npz 数据
        data = np.load(data_path, allow_pickle=True)
        self.load = data['load'].astype(np.float32)
        if self.load.ndim > 1: self.load = self.load.flatten()
        
        self.images = data['images'].astype(np.float32)
        raw_text = data['text'] # 所有原始文本
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 2. 初始化分词器
        print("正在加载文本分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        
        print("正在预处理所有文本 ...")
        # 批量编码所有文本
        encoded_text = self.tokenizer(
            raw_text.tolist(),
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        self.text_ids = encoded_text['input_ids'] # [Total_Len, 32]
        print("文本预处理完成。")
        # ========================================

        # 3. 归一化
        self.load_mean = np.mean(self.load)
        self.load_std = np.std(self.load)
        self.load_norm = (self.load - self.load_mean) / (self.load_std + 1e-5)
        
        self.n_samples = len(self.load) - (seq_len + pred_len) + 1
        print(f"数据集加载完成! 共有 {self.n_samples} 个样本。")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        # A. 负荷
        seq_x = self.load_norm[s_begin:s_end] 
        seq_y = self.load_norm[r_begin:r_end]
        
        # B. 图像
        img_x = self.images[s_begin:s_end]
        
        # C. 文本 (⚡现在直接切片取 Tensor，速度极快)
        text_x = self.text_ids[s_begin:s_end] 
        
        return {
            'x_load': torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1),
            'x_img': torch.tensor(img_x, dtype=torch.float32),
            'x_text': text_x, # 已经是 Tensor 了
            'y_load': torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1)
        }

# ================= 测试代码 =================
if __name__ == "__main__":
    # 实例化数据集
    dataset = MMTimeSeriesDataset('./processed_data_final.npz')
    
    # 建立 DataLoader (模拟训练时的批量读取)
    # batch_size=4: 一次取4个样本
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 取出一组看看形状
    batch = next(iter(dataloader))
    
    print("\n=== 数据形状检查 ===")
    print(f"Input Load Shape: {batch['x_load'].shape}") 
    # 预期: [4, 168, 1] -> [Batch, Time, Feature]
    
    print(f"Input Image Shape: {batch['x_img'].shape}") 
    # 预期: [4, 168, 2, 32, 32] -> [Batch, Time, Channel, H, W]
    
    print(f"Input Text Shape: {batch['x_text'].shape}") 
    # 预期: [4, 168, 32] -> [Batch, Time, Token_Len]
    
    print(f"Target Load Shape: {batch['y_load'].shape}") 
    # 预期: [4, 24, 1]