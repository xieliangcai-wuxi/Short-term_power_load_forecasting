import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
import logging
from tqdm import tqdm

# 导入自定义模块
from model.MMLSTMModel import MMLSTMModel 
from dataset import MMTimeSeriesDataset

# ==========================================
# 1. 顶刊级日志与环境配置
# ==========================================
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = f'./log/lstm_experiment_{TIMESTAMP}'
MODEL_DIR = os.path.join(LOG_ROOT, 'models')
FIGURE_DIR = os.path.join(LOG_ROOT, 'figures')
DATA_RECORD_DIR = os.path.join(LOG_ROOT, 'data_records')

for path in [MODEL_DIR, FIGURE_DIR, DATA_RECORD_DIR]:
    os.makedirs(path, exist_ok=True)

# 配置 Logger
logger = logging.getLogger('MM-LSTM')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 文件日志
file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'training_log.txt'), encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台日志 (通过 Logger 重定向，替代 print)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 全局配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50 
PATIENCE = 7 
DATA_PATH = './process_data/processed_data_10years_v2.npz' 
SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_lstm.pth')

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. 物理反归一化与指标工具
# ==========================================
class MetricTracker:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def inverse_transform(self, y_norm):
        if torch.is_tensor(y_norm):
            y_norm = y_norm.cpu().detach().numpy()
        return y_norm * self.std + self.mean

    def calculate_metrics(self, y_true_real, y_pred_real):
        # 物理量级计算 (IEEE 标准)
        mae = np.mean(np.abs(y_pred_real - y_true_real))
        rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
        mape = np.mean(np.abs((y_pred_real - y_true_real) / y_true_real)) * 100
        return rmse, mae, mape

# ==========================================
# 3. 训练核心组件
# ==========================================
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
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        logger.info(f'   [CheckPoint] Val Loss improved. Saving model...')
        torch.save(model.state_dict(), path)

# ==========================================
# 4. 主程序流程
# ==========================================
if __name__ == "__main__":
    seed_everything(50)
    logger.info("=" * 60)
    logger.info("MM-LSTM Multi-modal Forecasting Pipeline | IEEE Standard")
    logger.info(f"Start Time: {datetime.datetime.now()}")
    logger.info(f"Using Device: {DEVICE}")
    logger.info("=" * 60)

    # A. 数据加载与时间维度审计
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        sys.exit()

    # 加载数据集并打印严谨的时空信息
    train_set = MMTimeSeriesDataset(DATA_PATH, mode='train')
    val_set = MMTimeSeriesDataset(DATA_PATH, mode='val')
    test_set = MMTimeSeriesDataset(DATA_PATH, mode='test')

    # 获取 npz 里的原始时间戳进行审计 (假设 dataset 已经包含了这些 raw 数组)
    # 这在论文中是证明实验严谨性（非重叠）的关键
    raw_data = np.load(DATA_PATH, allow_pickle=True)
    times = raw_data['times']
    total_len = len(times)
    
    # 打印时间维度信息 (对应 dataset 内部 80/10/10 的逻辑)
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)
    
    logger.info(">>> Dataset Temporal & Statistical Audit <<<")
    logger.info(f"Train Segment: {times[0]} to {times[train_end-1]} | Mean: {train_set.load_mean:.2f} | Std: {train_set.load_std:.2f}")
    logger.info(f"Val Segment:   {times[train_end]} to {times[val_end-1]} | Mean: {val_set.load_mean:.2f} | Std: {val_set.load_std:.2f}")
    logger.info(f"Test Segment:  {times[val_end]} to {times[-1]} | Mean: {test_set.load_mean:.2f} | Std: {test_set.load_std:.2f}")
    logger.info(f"Total alignment length: {total_len} hourly samples.")
    logger.info("=" * 60)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化追踪器 (修复之前的属性错误)
    tracker = MetricTracker(train_set.load_mean, train_set.load_std)

    # B. 模型初始化
    model = MMLSTMModel(seq_len=168, pred_len=24).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    # C. 训练循环
    logger.info("\n>>> Starting Training Loop <<<")
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)
        
        for batch in pbar:
            x_load = batch['x_load'].to(DEVICE)
            x_img = batch['x_img'].to(DEVICE)
            x_text = batch['x_text'].to(DEVICE)
            y_true = batch['y_load'].to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(x_load, x_img, x_text) 
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix(train_mse=loss.item())

        # D. 验证环节
        model.eval()
        epoch_val_loss = 0
        all_val_preds, all_val_trues = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x_load, x_img, x_text, y_true = batch['x_load'].to(DEVICE), batch['x_img'].to(DEVICE), batch['x_text'].to(DEVICE), batch['y_load'].to(DEVICE)
                y_pred = model(x_load, x_img, x_text)
                epoch_val_loss += criterion(y_pred, y_true).item()
                
                all_val_preds.append(tracker.inverse_transform(y_pred))
                all_val_trues.append(tracker.inverse_transform(y_true))

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        val_preds_mw = np.concatenate(all_val_preds, axis=0)
        val_trues_mw = np.concatenate(all_val_trues, axis=0)
        _, _, v_mape = tracker.calculate_metrics(val_trues_mw, val_preds_mw)

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Summary: Train_MSE={avg_train_loss:.6f} | Val_MSE={avg_val_loss:.6f} | Val_MAPE={v_mape:.2f}%")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(v_mape)

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, SAVE_PATH)
        if early_stopping.early_stop: 
            logger.info("Early stopping triggered.")
            break

    # E. 最终测试评估
    logger.info("\n" + "="*20 + " Final Test Performance " + "="*20)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    
    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Testing"):
            x_load, x_img, x_text, y_true = batch['x_load'].to(DEVICE), batch['x_img'].to(DEVICE), batch['x_text'].to(DEVICE), batch['y_load'].to(DEVICE)
            y_pred = model(x_load, x_img, x_text)
            test_preds.append(tracker.inverse_transform(y_pred))
            test_trues.append(tracker.inverse_transform(y_true))

    final_preds = np.squeeze(np.concatenate(test_preds, axis=0))
    final_trues = np.squeeze(np.concatenate(test_trues, axis=0))

    rmse, mae, mape = tracker.calculate_metrics(final_trues, final_preds)
    logger.info(f"Test RMSE: {rmse:.2f} MW")
    logger.info(f"Test MAE:  {mae:.2f} MW")
    logger.info(f"Test MAPE: {mape:.2f}%")

    # F. 持久化与绘图
    np.savez(os.path.join(DATA_RECORD_DIR, 'test_results.npz'), preds=final_preds, trues=final_trues)
    
    plt.switch_backend('Agg')
    plt.figure(figsize=(12, 6))
    plt.plot(final_trues[0], label='Actual Load', color='black', alpha=0.8)
    plt.plot(final_preds[0], label='Predicted (MM-LSTM)', color='red', linestyle='--')
    plt.title(f"24-Hour Horizon Forecasting | MAPE: {mape:.2f}%")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, 'test_sample_compare.png'), dpi=300)
    
    logger.info(f"\nExperiment complete. Data saved in: {LOG_ROOT}")