# -*- coding: utf-8 -*-
"""
MM-GRU 多模态电力负荷预测训练流程 (Bi-GRU + Patching + Attention)
核心优势：利用 GRU 的时序归纳偏置解决 Transformer 在小数据集上的过拟合问题
适配数据：已修复夏令时错位的 UTC 对齐数据
"""
import os
import sys
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 路径兼容导入 =====================
try:
    from model.MMGRUModel import MMGRUModel
    from dataset import MMTimeSeriesDataset
except ImportError:
    from MMGRUModel import MMGRUModel
    from dataset import MMTimeSeriesDataset

# ===================== 全局配置 =====================
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = f'./log/gru_experiment_{TIMESTAMP}' # 区分实验目录

MODEL_DIR = os.path.join(LOG_ROOT, 'models')
FIGURE_DIR = os.path.join(LOG_ROOT, 'figures')
DATA_RECORD_DIR = os.path.join(LOG_ROOT, 'data_records')

for path in [MODEL_DIR, FIGURE_DIR, DATA_RECORD_DIR]:
    os.makedirs(path, exist_ok=True)

# 日志配置
logger = logging.getLogger('MM-GRU')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'training_log.txt'), encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 超参数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50
PATIENCE = 10  # GRU 收敛稍慢但更稳，给它多一点耐心
DATA_PATH = './processed_data_final.npz' # 【重要】指向刚才生成的修复版数据

SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_gru.pth')
HISTORY_SAVE_PATH = os.path.join(DATA_RECORD_DIR, 'training_history.npz')
PLOT_RESULT_PATH = os.path.join(FIGURE_DIR, 'gru_forecast_result.png')
PLOT_LOSS_PATH = os.path.join(FIGURE_DIR, 'gru_loss_curve.png')

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 工具类 (复用你的优秀代码) =====================
class MetricTracker:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def inverse_transform(self, y_norm):
        if torch.is_tensor(y_norm): y_norm = y_norm.cpu().detach().numpy()
        return y_norm * self.std + self.mean
    def calculate_metrics(self, y_true_real, y_pred_real):
        mae = np.mean(np.abs(y_pred_real - y_true_real))
        rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_pred_real - y_true_real) / y_true_real)) * 100
        return rmse, mae, mape

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.delta = delta
    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'   [EarlyStop] Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience: self.early_stop = True
    def save_checkpoint(self, model, path):
        logger.info(f'   [CheckPoint] Val Loss improved. Saving best MM-GRU model...')
        torch.save(model.state_dict(), path)

# ===================== 主程序 =====================
if __name__ == "__main__":
    seed_everything(50) # 与之前保持一致
    logger.info("=" * 60)
    logger.info("MM-GRU Forecasting Pipeline (Data-Fixed Version)")
    logger.info("=" * 60)

    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}. Please run process_data_final.py first!")
        sys.exit()

    # 加载数据集
    train_set = MMTimeSeriesDataset(DATA_PATH, mode='train')
    val_set = MMTimeSeriesDataset(DATA_PATH, mode='val')
    test_set = MMTimeSeriesDataset(DATA_PATH, mode='test')

    logger.info(">>> Dataset Audit <<<")
    logger.info(f"Train Samples: {len(train_set)} | Mean: {train_set.load_mean:.2f}")
    logger.info(f"Val Samples:   {len(val_set)}")
    logger.info(f"Test Samples:  {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    tracker = MetricTracker(train_set.load_mean, train_set.load_std)

    # 初始化 Bi-GRU 模型
    # d_model=64, n_layers=2 (双向后内部维度翻倍，刚好适合 PJM 数据规模)
    model = MMGRUModel(seq_len=168, pred_len=24, d_model=64, n_layers=2).to(DEVICE)
    
    criterion = nn.MSELoss()
    
    # 【关键】使用 AdamW + 较强的 Weight Decay (1e-3)
    # GRU 结构简单，配合强正则化，能逼出它的泛化潜力
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    logger.info("\n>>> Starting Training Loop (Bi-GRU) <<<")
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', file=sys.stdout)
        
        for batch in pbar:
            x_load = batch['x_load'].to(DEVICE, non_blocking=True)
            x_img = batch['x_img'].to(DEVICE, non_blocking=True)
            x_text = batch['x_text'].to(DEVICE, non_blocking=True)
            y_true = batch['y_load'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            y_pred = model(x_load, x_img, x_text)
            
            # 维度对齐检查
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1: y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1: y_true = y_true.squeeze(-1)

            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix(train_mse=loss.item())

        # 验证流程
        model.eval()
        epoch_val_loss = 0
        all_val_preds, all_val_trues = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x_load = batch['x_load'].to(DEVICE)
                x_img = batch['x_img'].to(DEVICE)
                x_text = batch['x_text'].to(DEVICE)
                y_true = batch['y_load'].to(DEVICE)
                
                y_pred = model(x_load, x_img, x_text)
                
                if y_pred.shape != y_true.shape:
                    if y_pred.dim() == 3 and y_pred.shape[-1] == 1: y_pred = y_pred.squeeze(-1)
                    elif y_true.dim() == 3 and y_true.shape[-1] == 1: y_true = y_true.squeeze(-1)

                epoch_val_loss += criterion(y_pred, y_true).item()
                all_val_preds.append(tracker.inverse_transform(y_pred))
                all_val_trues.append(tracker.inverse_transform(y_true))
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        val_preds_mw = np.concatenate(all_val_preds, axis=0)
        val_trues_mw = np.concatenate(all_val_trues, axis=0)
        _, _, v_mape = tracker.calculate_metrics(val_trues_mw, val_preds_mw)
        
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val MAPE: {v_mape:.2f}%")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(v_mape)

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, SAVE_PATH)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    # 保存历史
    np.savez(HISTORY_SAVE_PATH, train_loss=history['train_loss'], val_loss=history['val_loss'], val_mape=history['val_mape'])

    # 最终测试
    logger.info("\n" + "="*20 + " Final Test " + "="*20)
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True)) # 修复警告
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Testing"):
            x_load = batch['x_load'].to(DEVICE)
            x_img = batch['x_img'].to(DEVICE)
            x_text = batch['x_text'].to(DEVICE)
            y_true = batch['y_load'].to(DEVICE)
            
            y_pred = model(x_load, x_img, x_text)
            
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1: y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1: y_true = y_true.squeeze(-1)

            test_preds.append(tracker.inverse_transform(y_pred))
            test_trues.append(tracker.inverse_transform(y_true))

    final_preds = np.squeeze(np.concatenate(test_preds, axis=0))
    final_trues = np.squeeze(np.concatenate(test_trues, axis=0))
    np.savez(os.path.join(DATA_RECORD_DIR, 'test_results.npz'), preds=final_preds, trues=final_trues)
    
    rmse, mae, mape = tracker.calculate_metrics(final_trues, final_preds)
    logger.info(f"Test RMSE: {rmse:.2f} MW")
    logger.info(f"Test MAE:  {mae:.2f} MW")
    logger.info(f"Test MAPE: {mape:.2f}%")

    # 绘图逻辑 (略，已自动保存)
    logger.info(f"Experiment Done. Results in {LOG_ROOT}")