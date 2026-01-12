import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os

class MMTimeSeriesDataset(Dataset):
    def __init__(self, data_path, seq_len=168, pred_len=24, mode='train'):
        """
        mode: 'train', 'val', 'test'
        """
        # 1. 读取 .npz 数据
        data = np.load(data_path, allow_pickle=True)
        raw_load = data['load'].astype(np.float32)
        raw_images = data['images'].astype(np.float32)
        raw_text_list = data['text']
        
        # 获取我们之前存入的训练集标尺 (Anti-Leakage)
        meta = data['meta'].item()
        self.load_mean = meta['train_mean']
        self.load_std = meta['train_std']
        
        # 2. 严格的时间轴切分 (必须在归一化前确定范围)
        total_len = len(raw_load)
        train_end = int(total_len * 0.8)
        val_end = int(total_len * 0.9)
        
        if mode == 'train':
            start_p, end_p = 0, train_end
        elif mode == 'val':
            start_p, end_p = train_end, val_end
        else: # test
            start_p, end_p = val_end, total_len
            
        # 截取对应片段
        self.load = raw_load[start_p:end_p]
        self.images = raw_images[start_p:end_p]
        self.mode_text = raw_text_list[start_p:end_p]
        
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 3. 初始化分词器并预处理文本
        # 注意：为了节省内存和时间，我们只处理当前 mode 需要的文本
        print(f"正在加载 [{mode}] 文本分词器及预处理...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        encoded_text = self.tokenizer(
            self.mode_text.tolist(),
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        self.text_ids = encoded_text['input_ids'] 

        # 4. 使用【训练集标尺】进行归一化 (核心严谨点)
        self.load_norm = (self.load - self.load_mean) / (self.load_std + 1e-5)
        
        # 计算该模式下可生成的滑动窗口数量
        self.n_samples = len(self.load) - (seq_len + pred_len) + 1
        print(f"[{mode}] 数据集加载完成! 共有 {self.n_samples} 个样本。")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        # A. 负荷 [seq_len, 1]
        seq_x = self.load_norm[s_begin:s_end] 
        seq_y = self.load_norm[r_begin:r_end]
        
        # B. 图像序列 [seq_len, 2, 32, 32] -> 适配方案 A
        # 这里取完整的 168 步图像
        img_x = self.images[s_begin:s_end]
        
        # C. 文本序列 [seq_len, 32]
        text_x = self.text_ids[s_begin:s_end] 
        
        return {
            'x_load': torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1),
            'x_img': torch.tensor(img_x, dtype=torch.float32),
            'x_text': text_x, 
            'y_load': torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1)
        }