import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ==========================================
# 1. 修正导入路径
# ==========================================
# 确保你的 Mamba 模型保存在 model/MMMambaModel.py 中
try:
    from model.MMMambaModel import MMMambaModel
except ImportError:
    # 兼容处理
    import model.MMMambaModel as MMMambaModel

from dataset import MMTimeSeriesDataset

# ================= 配置参数 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50 
PATIENCE = 5 
DATA_PATH = './process_data/processed_data_10years.npz' 

# --- 改进：统一文件保存路径 (保持一致性) ---
MODEL_DIR = './best_model_file'
FIGURE_DIR = './figure'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_mamba.pth')
PLOT_RESULT_PATH = os.path.join(FIGURE_DIR, 'mamba_forecast_result.png')
PLOT_LOSS_PATH = os.path.join(FIGURE_DIR, 'mamba_loss_curve.png')

# ================= 工具函数 =================

def inverse_transform(y_norm, stats):
    if torch.is_tensor(y_norm):
        y_norm = y_norm.cpu().detach().numpy()
    mean = stats['mean']
    std = stats['std']
    y_real = y_norm * (std + 1e-5) + mean
    return y_real

def calculate_metrics_real(y_true_real, y_pred_real):
    mae = np.mean(np.abs(y_pred_real - y_true_real))
    rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
    mape = np.mean(np.abs((y_pred_real - y_true_real) / (y_true_real + 1.0))) * 100
    return rmse, mae, mape

# ================= 早停机制 =================
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

# ================= 主程序 =================

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
    # 2. 修正随机种子 (统一为 42)
    seed_everything(42)

    print(f"Using device: {DEVICE}")
    print(f"Model will be saved to: {SAVE_PATH}")

    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件 {DATA_PATH}")
        exit()
    
    dataset = MMTimeSeriesDataset(DATA_PATH)
    total_len = len(dataset)
    
    stats = {'mean': dataset.load_mean, 'std': dataset.load_std}
    print(f"数据统计量: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")

    train_size = int(total_len * 0.7)
    val_size = int(total_len * 0.1)
    test_size = total_len - train_size - val_size

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total_len))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # --- 模型初始化 ---
    try:
        model = MMMambaModel(seq_len=168, pred_len=24).to(DEVICE)
    except NameError:
        model = MMMambaModel.MMMambaModel(seq_len=168, pred_len=24).to(DEVICE)

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    print("\n=== 开始训练 Mamba 模型 ===")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for batch in loop:
            x_load = batch['x_load'].to(DEVICE, non_blocking=True)
            x_img = batch['x_img'].to(DEVICE, non_blocking=True)
            x_text = batch['x_text'].to(DEVICE, non_blocking=True)
            
            # 3. 修正：删除了 .squeeze(-1)
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
                
                # 3. 修正：删除了 .squeeze(-1)
                y_true = batch['y_load'].to(DEVICE)
                
                y_pred = model(x_load, x_img, x_text)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                
                val_preds_real.append(inverse_transform(y_pred, stats))
                val_trues_real.append(inverse_transform(y_true, stats))
        
        avg_val_loss = val_loss / len(val_loader)
        
        val_preds_real = np.concatenate(val_preds_real, axis=0)
        val_trues_real = np.concatenate(val_trues_real, axis=0)
        _, _, val_mape = calculate_metrics_real(val_trues_real, val_preds_real)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | \033[92mVal MAPE: {val_mape:.2f}%\033[0m")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(val_mape)

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, SAVE_PATH)
        if early_stopping.early_stop:
            print("早停触发！停止训练。")
            break

    # --- E. 最终测试 ---
    print("\n=== 最终测试 (Test Set) ===")
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    test_preds_real = []
    test_trues_real = []

    print("正在进行最终预测...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x_load = batch['x_load'].to(DEVICE)
            x_img = batch['x_img'].to(DEVICE)
            x_text = batch['x_text'].to(DEVICE)
            
            # 3. 修正：删除了 .squeeze(-1)
            y_true = batch['y_load'].to(DEVICE)
            
            y_pred = model(x_load, x_img, x_text)
            test_preds_real.append(inverse_transform(y_pred, stats))
            test_trues_real.append(inverse_transform(y_true, stats))

    preds = np.concatenate(test_preds_real, axis=0)
    trues = np.concatenate(test_trues_real, axis=0)

    rmse, mae, mape = calculate_metrics_real(trues, preds)

    print("-" * 35)
    print(f"Mamba 测试集最终结果:")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
    print(f"MAPE: {mape:.2f} %")
    print("-" * 35)

    # 绘图
    sample_mapes = np.mean(np.abs((preds - trues) / (trues + 1.0)), axis=1) * 100
    best_idx = np.argmin(sample_mapes)
    worst_idx = np.argmax(sample_mapes)
    random_idx = np.random.randint(0, len(preds))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(trues[best_idx], label='Truth', color='black')
    plt.plot(preds[best_idx], label='Pred (Mamba)', color='green', linestyle='--')
    plt.title(f'Mamba Best Case (MAPE: {sample_mapes[best_idx]:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(trues[random_idx], label='Truth', color='black')
    plt.plot(preds[random_idx], label='Pred (Mamba)', color='blue', linestyle='--')
    plt.title(f'Mamba Random Case (MAPE: {sample_mapes[random_idx]:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(trues[worst_idx], label='Truth', color='black')
    plt.plot(preds[worst_idx], label='Pred (Mamba)', color='red', linestyle='--')
    plt.title(f'Mamba Worst Case (MAPE: {sample_mapes[worst_idx]:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"正在保存预测对比图到: {PLOT_RESULT_PATH}")
    plt.savefig(PLOT_RESULT_PATH)
    plt.show()
    plt.close()
    
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Mamba Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    print(f"正在保存 Loss 曲线到: {PLOT_LOSS_PATH}")
    plt.savefig(PLOT_LOSS_PATH)
    plt.show()
    plt.close()