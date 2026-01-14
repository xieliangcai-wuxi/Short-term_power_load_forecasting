import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os

class MMTimeSeriesDataset(Dataset):
    def __init__(self, data_path, seq_len=168, pred_len=24, mode='train'):
        """
        mode: 'train', 'val', 'test'
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode

        # 1. 读取 .npz 数据
        data = np.load(data_path, allow_pickle=True)
        raw_load = data['load'].astype(np.float32)
        raw_images = data['images'].astype(np.float32)
        raw_text_list = data['text']
        
        # 获取训练集标尺
        meta = data['meta'].item()
        self.load_mean = meta['train_mean']
        self.load_std = meta['train_std']
        
        # 2. 带有“回溯缓冲”的时间轴切分 (关键修改)
        total_len = len(raw_load)
        train_end = int(total_len * 0.8)
        val_end = int(total_len * 0.9)
        
        if mode == 'train':
            # 训练集从头开始
            start_p = 0
            end_p = train_end
        elif mode == 'val':
            # [修改] 验证集需要往回多取 seq_len 长度，作为第一个样本的输入 x
            start_p = train_end - seq_len 
            end_p = val_end
        else: # test
            # [修改] 测试集同理
            start_p = val_end - seq_len
            end_p = total_len
            
        # 截取对应片段 (包含了缓冲区)
        self.load = raw_load[start_p:end_p]
        self.images = raw_images[start_p:end_p]
        self.mode_text = raw_text_list[start_p:end_p]
        
        # 3. 初始化分词器 (增加网络容错)
        # print(f"正在加载 [{mode}] 文本分词器...")
        try:
            # 优先读本地缓存，解决服务器连不上 HuggingFace 的问题
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        encoded_text = self.tokenizer(
            self.mode_text.tolist(),
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        self.text_ids = encoded_text['input_ids'] 

        # 4. 归一化 (使用训练集标尺)
        self.load_norm = (self.load - self.load_mean) / (self.load_std + 1e-5)
        
        # 5. 计算样本数量
        # 因为我们多取了 seq_len 长度的缓冲，所以这里的计算逻辑不需要变
        # 现在的 self.load 长度 = (原长度 + seq_len)
        # 所以能生成的样本数正好覆盖了 原长度 的所有预测点
        self.n_samples = len(self.load) - (seq_len + pred_len) + 1
        
        # 边界检查 (防止数据太少报错)
        if self.n_samples <= 0:
             raise ValueError(f"[{mode}] 数据集过短，无法生成 seq_len={seq_len} 的样本，请检查切分比例。")

        print(f"[{mode}] 数据集就绪 | 范围: {start_p}->{end_p} | 样本数: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 这里的 idx 是从 0 开始的
        # 因为我们在 __init__ 里已经加上了 buffer，所以直接切片即可
        
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        # A. 负荷
        seq_x = self.load_norm[s_begin:s_end] 
        seq_y = self.load_norm[r_begin:r_end]
        
        # B. 图像
        img_x = self.images[s_begin:s_end]
        
        # C. 文本
        text_x = self.text_ids[s_begin:s_end] 
        
        return {
            'x_load': torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1),
            'x_img': torch.tensor(img_x, dtype=torch.float32),
            'x_text': text_x, 
            'y_load': torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1)
        }