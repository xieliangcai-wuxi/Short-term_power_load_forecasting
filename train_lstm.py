import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
from tqdm import tqdm

# 假设你的目录结构允许这样导入
import model.MMLSTMModel as MMLSTMModel 
from dataset import MMTimeSeriesDataset

# ================= 0. 日志与配置系统 (关键修改) =================

# --- 基础配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50 
PATIENCE = 5 
DATA_PATH = './process_data/processed_data_10years_v2.npz' 

# --- 目录管理：所有内容统一保存到 ./log ---
# 为了区分每次实验，建议加上时间戳，或者固定文件夹名称
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = f'./log/lstm_experiment_{TIMESTAMP}' # 或者直接用 './log/lstm_v1'

# 子文件夹路径
MODEL_DIR = os.path.join(LOG_ROOT, 'models')
FIGURE_DIR = os.path.join(LOG_ROOT, 'figures')
DATA_RECORD_DIR = os.path.join(LOG_ROOT, 'data_records')

# 自动创建文件夹
for path in [MODEL_DIR, FIGURE_DIR, DATA_RECORD_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# 具体文件路径
SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_lstm.pth')
LOG_TXT_PATH = os.path.join(LOG_ROOT, 'training_log.txt')
HISTORY_SAVE_PATH = os.path.join(DATA_RECORD_DIR, 'training_history.npz') # 保存Loss数据用于后续对比
PLOT_RESULT_PATH = os.path.join(FIGURE_DIR, 'lstm_forecast_result.png')
PLOT_LOSS_PATH = os.path.join(FIGURE_DIR, 'lstm_loss_curve.png')

# --- 双向日志记录器 (同时打印到控制台和文件) ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 立即写入，防止崩溃丢失
 
    def flush(self):
        # needed for python 3 compatibility
        pass

# 重定向 print
sys.stdout = Logger(LOG_TXT_PATH)

def inverse_transform(y_norm, stats):
    """反归一化: Real = Norm * Std + Mean"""
    if torch.is_tensor(y_norm):
        y_norm = y_norm.cpu().detach().numpy()
    
    mean = stats['mean']
    std = stats['std']
    y_real = y_norm * (std + 1e-5) + mean
    return y_real

def calculate_metrics_real(y_true_real, y_pred_real):
    """计算 RMSE, MAE, MAPE"""
    mae = np.mean(np.abs(y_pred_real - y_true_real))
    rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
    # 防止分母为0
    mape = np.mean(np.abs((y_pred_real - y_true_real) / (y_true_real + 1.0))) * 100
    return rmse, mae, mape

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'   [EarlyStop] 计数: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        print(f'   [CheckPoint] Val Loss 下降 ({self.best_loss:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), path)
        self.best_loss = val_loss
def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything(50)
    
    # 打印配置信息到日志
    print("=" * 30)
    print(f"Time: {datetime.datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"Log Root: {LOG_ROOT}")
    print("=" * 30)

    # --- A. 加载数据 ---
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件 {DATA_PATH}")
        exit()
    
    dataset = MMTimeSeriesDataset(DATA_PATH)
    total_len = len(dataset)
    stats = {'mean': dataset.load_mean, 'std': dataset.load_std}
    print(f"数据统计量: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")

    # --- B. 划分 ---
    train_size = int(total_len * 0.7)
    val_size = int(total_len * 0.1)
    
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total_len))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # --- C. 模型初始化 ---
    try:
        model = MMLSTMModel.MMLSTMModel(seq_len=168, pred_len=24).to(DEVICE)
    except AttributeError:
        model = MMLSTMModel(seq_len=168, pred_len=24).to(DEVICE)
        
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE)

    # 记录字典
    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    # --- D. 训练循环 ---
    print("\n=== 开始训练 LSTM 模型 ===")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        # Tqdm output directed to stderr, won't clutter the log file too much if handled well
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', file=sys.stderr)
        
        for batch in loop:
            x_load = batch['x_load'].to(DEVICE, non_blocking=True)
            x_img = batch['x_img'].to(DEVICE, non_blocking=True)
            x_text = batch['x_text'].to(DEVICE, non_blocking=True)
            y_true = batch['y_load'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            y_pred = model(x_load, x_img, x_text)
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_preds_real = []
        val_trues_real = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_load = batch['x_load'].to(DEVICE)
                x_img = batch['x_img'].to(DEVICE)
                x_text = batch['x_text'].to(DEVICE)
                y_true = batch['y_load'].to(DEVICE)
                
                y_pred = model(x_load, x_img, x_text)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                
                val_preds_real.append(inverse_transform(y_pred, stats))
                val_trues_real.append(inverse_transform(y_true, stats))
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 拼接并计算指标
        val_preds_real = np.concatenate(val_preds_real, axis=0)
        val_trues_real = np.concatenate(val_trues_real, axis=0)
        _, _, val_mape = calculate_metrics_real(val_trues_real, val_preds_real)
        
        # 记录并打印
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val MAPE: {val_mape:.2f}%")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(val_mape)

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, SAVE_PATH)
        
        if early_stopping.early_stop:
            print("早停触发！停止训练。")
            break

    # --- E. 保存训练数据 (重要：用于后续对比) ---
    print(f"\n正在保存训练历史数据到: {HISTORY_SAVE_PATH}")
    np.savez(HISTORY_SAVE_PATH, 
             train_loss=np.array(history['train_loss']),
             val_loss=np.array(history['val_loss']),
             val_mape=np.array(history['val_mape']))

    # --- F. 最终测试与绘图 ---
    print("\n=== 最终测试 (Test Set) ===")
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    test_preds_real = []
    test_trues_real = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", file=sys.stderr):
            x_load = batch['x_load'].to(DEVICE)
            x_img = batch['x_img'].to(DEVICE)
            x_text = batch['x_text'].to(DEVICE)
            y_true = batch['y_load'].to(DEVICE)
            y_pred = model(x_load, x_img, x_text)
            
            test_preds_real.append(inverse_transform(y_pred, stats))
            test_trues_real.append(inverse_transform(y_true, stats))

    preds = np.concatenate(test_preds_real, axis=0)
    trues = np.concatenate(test_trues_real, axis=0)
    preds = np.squeeze(preds)
    trues = np.squeeze(trues)

    # 保存预测结果（可选，为了后续画更精细的图）
    np.savez(os.path.join(DATA_RECORD_DIR, 'test_results.npz'), preds=preds, trues=trues)

    rmse, mae, mape = calculate_metrics_real(trues, preds)

    print("-" * 35)
    print(f"LSTM 测试集最终结果:")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
    print(f"MAPE: {mape:.2f} %")
    print("-" * 35)

    # === 绘图部分 ===
    # 切换 backend 防止服务器端报错
    plt.switch_backend('Agg') 

    # 1. 结果对比图
    sample_mapes = np.mean(np.abs((preds - trues) / (trues + 1.0)), axis=1) * 100
    best_idx = np.argmin(sample_mapes)
    worst_idx = np.argmax(sample_mapes)
    random_idx = np.random.randint(0, len(preds))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Best
    axes[0].plot(trues[best_idx], label='Truth', color='black', linewidth=1.5)
    axes[0].plot(preds[best_idx], label='LSTM', color='green', linestyle='--', linewidth=1.5)
    axes[0].set_title(f'Best Case (MAPE: {sample_mapes[best_idx]:.2f}%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Random
    axes[1].plot(trues[random_idx], label='Truth', color='black', linewidth=1.5)
    axes[1].plot(preds[random_idx], label='LSTM', color='blue', linestyle='--', linewidth=1.5)
    axes[1].set_title(f'Random Case (MAPE: {sample_mapes[random_idx]:.2f}%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Worst
    axes[2].plot(trues[worst_idx], label='Truth', color='black', linewidth=1.5)
    axes[2].plot(preds[worst_idx], label='LSTM', color='red', linestyle='--', linewidth=1.5)
    axes[2].set_title(f'Worst Case (MAPE: {sample_mapes[worst_idx]:.2f}%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_RESULT_PATH, dpi=300)
    print(f"预测对比图已保存: {PLOT_RESULT_PATH}")
    plt.close() 

    # 2. Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOT_LOSS_PATH, dpi=300)
    print(f"Loss 曲线已保存: {PLOT_LOSS_PATH}")
    plt.close()

    print("\n所有任务完成。日志和数据已保存至:", LOG_ROOT)