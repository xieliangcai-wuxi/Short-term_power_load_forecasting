import os
# --- Mamba 特有环境配置 (必须在 import torch 之前) ---
os.environ["MAMBA_FORCE_PYTHON"] = "1"  # 根据你的环境需求保留
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import logging
from tqdm import tqdm

# 导入自定义模块
try:
    from model.MMMambaModel import MMMambaModel
    from dataset import MMTimeSeriesDataset
except ImportError:
    # 路径兼容处理
    from MMMambaModel import MMMambaModel
    from dataset import MMTimeSeriesDataset

# ==========================================
# 1. 顶刊级日志与环境配置
# ==========================================
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 明确标注 mamba，方便实验管理
LOG_ROOT = f'./log/mamba_experiment_{TIMESTAMP}' 
MODEL_DIR = os.path.join(LOG_ROOT, 'models')
FIGURE_DIR = os.path.join(LOG_ROOT, 'figures')
DATA_RECORD_DIR = os.path.join(LOG_ROOT, 'data_records')

for path in [MODEL_DIR, FIGURE_DIR, DATA_RECORD_DIR]:
    os.makedirs(path, exist_ok=True)

# --- 配置标准 Logger ---
logger = logging.getLogger('MM-Mamba')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 文件日志
file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'training_log.txt'), encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- 全局参数 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50 
PATIENCE = 7 
# 注意：这里使用 v2 版本，确保包含 meta 信息
DATA_PATH = './process_data/processed_data_10years_v2.npz' 
SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_mamba.pth')

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
# 2. 物理反归一化与指标工具 (与 LSTM 保持一致)
# ==========================================
class MetricTracker:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def inverse_transform(self, y_norm):
        if torch.is_tensor(y_norm):
            y_norm = y_norm.cpu().detach().numpy()
        # 严格对齐归一化公式: Real = Norm * Std + Mean
        return y_norm * self.std + self.mean

    def calculate_metrics(self, y_true_real, y_pred_real):
        # 物理量级计算 (IEEE 标准)
        # 修正：移除了 +1.0，计算真实 MAPE
        mae = np.mean(np.abs(y_pred_real - y_true_real))
        rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
        
        # 处理分母极小值情况 (虽在负荷预测中罕见，但为了代码健壮性)
        with np.errstate(divide='ignore', invalid='ignore'):
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
    logger.info("MM-Mamba Multi-modal Forecasting Pipeline | IEEE Standard")
    logger.info(f"Start Time: {datetime.datetime.now()}")
    logger.info(f"Using Device: {DEVICE}")
    logger.info("=" * 60)

    # A. 数据加载与时间维度审计
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        sys.exit()

    # 加载数据集
    train_set = MMTimeSeriesDataset(DATA_PATH, mode='train')
    val_set = MMTimeSeriesDataset(DATA_PATH, mode='val')
    test_set = MMTimeSeriesDataset(DATA_PATH, mode='test')

    # --- 时序审计 (Temporal Audit) ---
    raw_data = np.load(DATA_PATH, allow_pickle=True)
    times = raw_data['times']
    total_len = len(times)
    
    # 对应 dataset 内部 80/10/10 的逻辑
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)
    
    logger.info(">>> Dataset Temporal & Statistical Audit <<<")
    logger.info(f"Train Segment: {times[0]} to {times[train_end-1]} | Mean: {train_set.load_mean:.2f} | Std: {train_set.load_std:.2f}")
    logger.info(f"Val Segment:   {times[train_end]} to {times[val_end-1]} | Mean: {val_set.load_mean:.2f} | Std: {val_set.load_std:.2f}")
    logger.info(f"Test Segment:  {times[val_end]} to {times[-1]} | Mean: {test_set.load_mean:.2f} | Std: {test_set.load_std:.2f}")
    logger.info("=" * 60)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化追踪器 (使用训练集标尺)
    tracker = MetricTracker(train_set.load_mean, train_set.load_std)

    # B. 模型初始化
    model = MMMambaModel(seq_len=168, pred_len=24).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) # 加上 weight_decay 防止过拟合
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    # C. 训练循环
    logger.info("\n>>> Starting Training Loop (Mamba) <<<")
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)
        
        for batch in pbar:
            x_load = batch['x_load'].to(DEVICE, non_blocking=True)
            x_img = batch['x_img'].to(DEVICE, non_blocking=True)
            x_text = batch['x_text'].to(DEVICE, non_blocking=True)
            y_true = batch['y_load'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            y_pred = model(x_load, x_img, x_text)
            
            # --- Mamba 维度对齐检查 ---
            # 如果 Mamba 输出是 (B, L) 而 y_true 是 (B, L, 1)，或者反之
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)

            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix(train_mse=loss.item())
        
        # D. 验证环节
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
                
                # 验证集同样的维度处理
                if y_pred.shape != y_true.shape:
                    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                        y_pred = y_pred.squeeze(-1)
                    elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                        y_true = y_true.squeeze(-1)
                
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
            x_load = batch['x_load'].to(DEVICE)
            x_img = batch['x_img'].to(DEVICE)
            x_text = batch['x_text'].to(DEVICE)
            y_true = batch['y_load'].to(DEVICE)
            
            y_pred = model(x_load, x_img, x_text)

            # 测试集维度处理
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)

            test_preds.append(tracker.inverse_transform(y_pred))
            test_trues.append(tracker.inverse_transform(y_true))

    final_preds = np.squeeze(np.concatenate(test_preds, axis=0))
    final_trues = np.squeeze(np.concatenate(test_trues, axis=0))

    # 持久化
    np.savez(os.path.join(DATA_RECORD_DIR, 'test_results.npz'), preds=final_preds, trues=final_trues)

    rmse, mae, mape = tracker.calculate_metrics(final_trues, final_preds)
    logger.info(f"Test RMSE: {rmse:.2f} MW")
    logger.info(f"Test MAE:  {mae:.2f} MW")
    logger.info(f"Test MAPE: {mape:.2f}%")

    # F. 绘图
    plt.switch_backend('Agg') 
    
    # 1. 随机/最佳/最差样本对比
    try:
        # 避免除以0，这里仅用于找样本索引
        sample_mapes = np.mean(np.abs((final_preds - final_trues) / (final_trues + 1e-5)), axis=1) * 100
        best_idx = np.argmin(sample_mapes)
        worst_idx = np.argmax(sample_mapes)
        random_idx = np.random.randint(0, len(final_preds))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ['Best Case', 'Random Case', 'Worst Case']
        indices = [best_idx, random_idx, worst_idx]
        colors = ['green', 'blue', 'red']

        for i, ax in enumerate(axes):
            idx = indices[i]
            ax.plot(final_trues[idx], label='Truth', color='black', linewidth=1.5)
            ax.plot(final_preds[idx], label='Mamba', color=colors[i], linestyle='--', linewidth=1.5)
            ax.set_title(f'{titles[i]} (MAPE: {sample_mapes[idx]:.2f}%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'mamba_forecast_result.png'), dpi=300)
    except Exception as e:
        logger.error(f"Plotting failed: {str(e)}")

    logger.info(f"\nExperiment complete. Logs saved in: {LOG_ROOT}")