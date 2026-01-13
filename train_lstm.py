# -*- coding: utf-8 -*-
"""
MM-LSTM 多模态时间序列负荷预测完整训练&测试流程
核心任务：基于负荷序列/图像特征/文本特征的多模态融合，实现24小时负荷预测
评估标准：IEEE标准回归指标 (RMSE/MAE/MAPE)，物理量纲为兆瓦(MW)
"""
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
from tqdm import tqdm  # 进度条可视化工具

# ===================== 自定义模块导入 & 路径兼容处理 =====================
# 兼容两种运行场景：项目根目录运行 / 当前文件目录直接运行
try:
    from model.MMLSTMModel import MMLSTMModel  # 项目规范目录结构
    from dataset import MMTimeSeriesDataset     # 多模态时序数据集类
except ImportError:
    # 路径兼容：当前文件与模型/数据集文件同目录时的导入方式
    from MMLSTMModel import MMLSTMModel
    from dataset import MMTimeSeriesDataset

# ===================== 全局实验目录 & 文件路径配置 =====================
# 生成时间戳，用于区分不同实验的结果文件，防止覆盖
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 实验根目录：按时间戳命名，归档所有日志/模型/图表/数据
LOG_ROOT = f'./log/lstm_experiment_{TIMESTAMP}'
MODEL_DIR = os.path.join(LOG_ROOT, 'models')    # 模型权重保存目录
FIGURE_DIR = os.path.join(LOG_ROOT, 'figures')  # 可视化图表保存目录
DATA_RECORD_DIR = os.path.join(LOG_ROOT, 'data_records')  # 训练历史/测试结果保存目录

# 递归创建目录，exist_ok=True：目录已存在时不报错
for path in [MODEL_DIR, FIGURE_DIR, DATA_RECORD_DIR]:
    os.makedirs(path, exist_ok=True)

# ===================== 日志系统配置 - 全局日志记录 =====================
# 日志器命名，区分其他模块日志
logger = logging.getLogger('MM-LSTM')
logger.setLevel(logging.INFO)  # 日志等级：只记录 INFO 及以上级别信息
# 日志格式：时间 + 日志等级 + 日志内容，标准化输出格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 1. 文件日志处理器：将日志写入文件，永久归档，编码utf-8防止中文乱码
file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'training_log.txt'), encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 2. 控制台日志处理器：将日志打印到终端，实时查看，替代原生print
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ===================== 全局超参数 & 常量配置（统一管理）=====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 硬件设备：优先使用GPU加速
BATCH_SIZE = 64          # 批次大小：显存充足可增大，过小会导致训练震荡
LR = 0.001               # 初始学习率
EPOCHS = 50              # 最大训练轮数
PATIENCE = 7             # 早停耐心值：验证集loss连续7轮不下降则停止训练
DATA_PATH = './process_data/processed_data_10years_v2.npz'  # 预处理后数据集路径

# 各类结果文件的保存路径（统一配置，便于修改）
SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_lstm.pth')          # 最优模型权重保存路径
HISTORY_SAVE_PATH = os.path.join(DATA_RECORD_DIR, 'training_history.npz')  # 训练历史保存路径
PLOT_RESULT_PATH = os.path.join(FIGURE_DIR, 'lstm_forecast_result.png')    # 预测结果可视化路径
PLOT_LOSS_PATH = os.path.join(FIGURE_DIR, 'lstm_loss_curve.png')           # 损失曲线可视化路径

# ===================== 全局工具函数 =====================
def seed_everything(seed=42):
    """
    固定全局随机种子，保证实验可复现性
    :param seed: 随机种子值，默认42
    """
    import random
    random.seed(seed)                # python原生随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    np.random.seed(seed)             # numpy随机种子
    torch.manual_seed(seed)          # CPU pytorch随机种子
    torch.cuda.manual_seed_all(seed) # 多GPU pytorch随机种子
    torch.backends.cudnn.deterministic = True  # 确定性卷积算法
    torch.backends.cudnn.benchmark = False     # 关闭卷积算法优化，保证一致性

# ==========================================
# 模块1：指标追踪器 - 反归一化+预测评估指标计算（IEEE标准）
# 核心功能：1.将归一化的预测值/真实值还原为物理量纲(MW) 2.计算回归任务评估指标
# ==========================================
class MetricTracker:
    def __init__(self, mean, std):
        """
        初始化指标追踪器
        :param mean: 训练集负荷数据的均值（反归一化用）
        :param std: 训练集负荷数据的标准差（反归一化用）
        """
        self.mean = mean
        self.std = std

    def inverse_transform(self, y_norm):
        """
        反归一化函数：将标准化后的数据 还原为 原始物理量纲(MW)
        标准化公式：y_norm = (y_real - mean) / std → 反归一化：y_real = y_norm * std + mean
        :param y_norm: 归一化后的预测值/真实值 (tensor/numpy)
        :return: 原始物理量纲的数值 (numpy数组)
        """
        # 兼容tensor张量输入，先转cpu→解除梯度→转numpy
        if torch.is_tensor(y_norm):
            y_norm = y_norm.cpu().detach().numpy()
        return y_norm * self.std + self.mean

    def calculate_metrics(self, y_true_real, y_pred_real):
        """
        计算回归任务三大评估指标（IEEE电力负荷预测标准指标，物理量纲统一为MW）
        :param y_true_real: 反归一化后的真实值 (numpy)
        :param y_pred_real: 反归一化后的预测值 (numpy)
        :return: rmse(均方根误差), mae(平均绝对误差), mape(平均绝对百分比误差)
        """
        # MAE：平均绝对误差，对异常值鲁棒，数值越小越好
        mae = np.mean(np.abs(y_pred_real - y_true_real))
        # RMSE：均方根误差，放大误差项，对大偏差惩罚更重，数值越小越好
        rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
        
        # MAPE：平均绝对百分比误差，百分比形式，直观反映预测精度，%越小越好
        # np.errstate：忽略除零/无效值警告，防止真实值为0时报错；+1e-5也可避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_pred_real - y_true_real) / y_true_real)) * 100
        return rmse, mae, mape

# ==========================================
# 模块2：早停机制 - 防止过拟合+保存最优模型（训练必备组件）
# 核心逻辑：当验证集损失连续N轮不再下降时，提前终止训练，同时保存最优模型权重
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        初始化早停器
        :param patience: 耐心值，连续多少轮验证集loss不下降则停止训练
        :param delta: 损失下降阈值，只有loss下降超过该值才算有效提升，防止微小波动
        """
        self.patience = patience    # 耐心值
        self.counter = 0            # 计数：当前连续未提升的轮数
        self.best_loss = np.inf     # 最优损失值：初始化为无穷大
        self.early_stop = False     # 早停标志位：是否触发早停
        self.delta = delta          # 损失提升阈值

    def __call__(self, val_loss, model, path):
        """
        每轮验证后调用，判断是否更新最优模型/触发早停
        :param val_loss: 当前轮次的验证集平均损失
        :param model: 当前训练的模型实例
        :param path: 最优模型权重的保存路径
        """
        # 情况1：当前验证损失 优于 最优损失-阈值 → 更新最优模型
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss          # 更新最优损失
            self.save_checkpoint(model, path)  # 保存当前最优模型权重
            self.counter = 0                   # 重置计数
        # 情况2：当前验证损失无有效提升 → 计数+1
        else:
            self.counter += 1
            logger.info(f'   [EarlyStop] Counter: {self.counter}/{self.patience}')
            # 计数达到耐心值 → 触发早停
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        """保存模型权重到指定路径"""
        logger.info(f'   [CheckPoint] Val Loss improved. Saving model...')
        torch.save(model.state_dict(), path)

# ==========================================
# 主程序入口 - 完整训练+验证+测试+可视化流程
# 执行逻辑：固定种子 → 日志初始化 → 数据加载 → 模型初始化 → 训练循环 → 测试评估 → 结果保存
# ==========================================
if __name__ == "__main__":
    # 1. 固定随机种子，保证实验可复现
    seed_everything(50)
    
    # 2. 打印实验启动信息，标准化日志头
    logger.info("=" * 60)
    logger.info("MM-LSTM Multi-modal Forecasting Pipeline | IEEE Standard Metrics")
    logger.info(f"Start Time: {datetime.datetime.now()}")
    logger.info(f"Using Device: {DEVICE}")
    logger.info("=" * 60)

    # ===================== 数据加载与时间维度审计 =====================
    # 校验数据集文件是否存在，不存在则终止程序并报错
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        sys.exit()

    # 加载多模态时序数据集，按训练/验证/测试模式划分（内部逻辑80%/10%/10%）
    train_set = MMTimeSeriesDataset(DATA_PATH, mode='train')
    val_set = MMTimeSeriesDataset(DATA_PATH, mode='val')
    test_set = MMTimeSeriesDataset(DATA_PATH, mode='test')

    # 加载原始数据文件，审计时间维度信息，确认数据集划分的时间区间
    raw_data = np.load(DATA_PATH, allow_pickle=True)
    times = raw_data['times']          # 原始时间戳数组
    total_len = len(times)             # 总时间序列长度（小时级样本）
    train_end = int(total_len * 0.8)   # 训练集结束位置
    val_end = int(total_len * 0.9)     # 验证集结束位置

    # 打印数据集审计日志：时间区间+统计特征，便于核对数据有效性
    logger.info(">>> Dataset Temporal & Statistical Audit <<<")
    logger.info(f"Train Segment: {times[0]} to {times[train_end-1]} | Mean: {train_set.load_mean:.2f} | Std: {train_set.load_std:.2f}")
    logger.info(f"Val Segment:   {times[train_end]} to {times[val_end-1]} | Mean: {val_set.load_mean:.2f} | Std: {val_set.load_std:.2f}")
    logger.info(f"Test Segment:  {times[val_end]} to {times[-1]} | Mean: {test_set.load_mean:.2f} | Std: {test_set.load_std:.2f}")
    logger.info(f"Total alignment length: {total_len} hourly samples.")
    logger.info("=" * 60)

    # 构建数据加载器：批处理+洗牌+显存优化
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化指标追踪器：传入训练集的均值和标准差，用于后续反归一化和指标计算
    tracker = MetricTracker(train_set.load_mean, train_set.load_std)

    # ===================== 模型与训练组件初始化 =====================
    # 初始化多模态LSTM模型：输入序列长度168h(7天)，预测序列长度24h(1天)，部署到指定设备
    model = MMLSTMModel(seq_len=168, pred_len=24).to(DEVICE)
    criterion = nn.MSELoss()  # 损失函数：均方误差，回归任务首选
    # 优化器：Adam，带权重衰减(1e-5)防止过拟合，学习率LR
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    # 学习率调度器：验证集loss不下降时，学习率减半，耐心值3轮
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # 早停器实例化
    early_stopping = EarlyStopping(patience=PATIENCE)

    # 训练历史记录字典：保存每轮的训练损失、验证损失、验证MAPE，用于后续可视化
    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    # ===================== 核心训练循环 =====================
    logger.info("\n>>> Starting Training Loop <<<")
    for epoch in range(EPOCHS):
        # 模型切换为训练模式：启用Dropout/BatchNorm等训练层
        model.train()
        epoch_train_loss = 0.0  # 累计当前轮次的训练损失
        
        # 进度条包装训练加载器，实时显示训练进度和当前批次损失
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)
        
        # 批次训练循环
        for batch in pbar:
            # 加载批次数据，分模态获取输入+真实值，部署到指定设备
            x_load = batch['x_load'].to(DEVICE)  # 负荷序列特征
            x_img = batch['x_img'].to(DEVICE)    # 图像模态特征
            x_text = batch['x_text'].to(DEVICE)  # 文本模态特征
            y_true = batch['y_load'].to(DEVICE)  # 真实负荷值（标签）
            
            optimizer.zero_grad()        # 清空梯度缓存，防止梯度累加
            y_pred = model(x_load, x_img, x_text)  # 模型前向传播，输出预测值
            loss = criterion(y_pred, y_true)       # 计算批次损失
            loss.backward()              # 反向传播，计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()             # 梯度更新，优化模型参数
            
            epoch_train_loss += loss.item()  # 累计批次损失
            pbar.set_postfix(train_mse=loss.item())  # 进度条实时显示当前批次损失

        # ===================== 每轮训练后 - 验证环节 =====================
        model.eval()  # 模型切换为评估模式：关闭Dropout/BatchNorm，固定参数
        epoch_val_loss = 0.0  # 累计当前轮次的验证损失
        all_val_preds, all_val_trues = [], []  # 保存所有验证集的预测值和真实值
        
        # 验证阶段关闭梯度计算：节省显存+加速计算，无反向传播
        with torch.no_grad():
            for batch in val_loader:
                # 加载验证批次数据
                x_load, x_img, x_text, y_true = batch['x_load'].to(DEVICE), batch['x_img'].to(DEVICE), batch['x_text'].to(DEVICE), batch['y_load'].to(DEVICE)
                y_pred = model(x_load, x_img, x_text)  # 模型前向传播
                epoch_val_loss += criterion(y_pred, y_true).item()  # 累计验证损失
                
                # 反归一化后保存，用于计算物理量纲的评估指标
                all_val_preds.append(tracker.inverse_transform(y_pred))
                all_val_trues.append(tracker.inverse_transform(y_true))

        # 计算当前轮次的平均训练/验证损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # 拼接所有验证集预测值和真实值，计算评估指标
        val_preds_mw = np.concatenate(all_val_preds, axis=0)
        val_trues_mw = np.concatenate(all_val_trues, axis=0)
        _, _, v_mape = tracker.calculate_metrics(val_trues_mw, val_preds_mw)

        # 打印本轮训练总结日志
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Summary: Train_MSE={avg_train_loss:.6f} | Val_MSE={avg_val_loss:.6f} | Val_MAPE={v_mape:.2f}%")
        
        # 保存本轮训练历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(v_mape)

        # 学习率调度器更新：根据验证损失调整学习率
        scheduler.step(avg_val_loss)
        # 早停器判断：是否更新最优模型/触发早停
        early_stopping(avg_val_loss, model, SAVE_PATH)
        if early_stopping.early_stop: 
            logger.info("Early stopping triggered. Training terminated early.")
            break

    # ===================== 训练完成 - 保存训练历史 =====================
    logger.info(f"\nSaving training history to: {HISTORY_SAVE_PATH}")
    np.savez(HISTORY_SAVE_PATH, 
             train_loss=np.array(history['train_loss']),
             val_loss=np.array(history['val_loss']),
             val_mape=np.array(history['val_mape']))

    # ===================== 最终测试评估（加载最优模型）=====================
    logger.info("\n" + "="*20 + " Final Test Performance Evaluation " + "="*20)
    # 加载训练过程中保存的最优模型权重
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()  # 模型切换为评估模式
    
    test_preds, test_trues = [], []  # 保存所有测试集的预测值和真实值
    # 测试阶段关闭梯度计算
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Testing Process"):
            x_load, x_img, x_text, y_true = batch['x_load'].to(DEVICE), batch['x_img'].to(DEVICE), batch['x_text'].to(DEVICE), batch['y_load'].to(DEVICE)
            y_pred = model(x_load, x_img, x_text)
            # 反归一化后保存
            test_preds.append(tracker.inverse_transform(y_pred))
            test_trues.append(tracker.inverse_transform(y_true))

    # 拼接所有测试集结果，压缩维度（去除冗余维度）
    final_preds = np.squeeze(np.concatenate(test_preds, axis=0))
    final_trues = np.squeeze(np.concatenate(test_trues, axis=0))

    # 计算测试集最终评估指标
    rmse, mae, mape = tracker.calculate_metrics(final_trues, final_preds)
    logger.info(f"Test Set Final Metrics (IEEE Standard) >>")
    logger.info(f"Test RMSE: {rmse:.2f} MW")
    logger.info(f"Test MAE:  {mae:.2f} MW")
    logger.info(f"Test MAPE: {mape:.2f}%")

    # ===================== 结果持久化 & 可视化绘图 =====================
    # 保存测试集预测结果和真实值，便于后续分析
    np.savez(os.path.join(DATA_RECORD_DIR, 'test_results.npz'), preds=final_preds, trues=final_trues)
    
    # 切换matplotlib后端为Agg：无GUI环境下（如服务器）可正常绘图保存，不报错
    plt.switch_backend('Agg')
    
    try:
        # 绘图1：预测结果对比图 → 最优/随机/最差样本 三图合一
        # 计算每个样本的MAPE，筛选最优/最差样本索引
        sample_mapes = np.mean(np.abs((final_preds - final_trues) / (final_trues + 1e-5)), axis=1) * 100
        best_idx = np.argmin(sample_mapes)   # MAPE最小→最优样本
        worst_idx = np.argmax(sample_mapes)  # MAPE最大→最差样本
        random_idx = np.random.randint(0, len(final_preds))  # 随机样本

        # 创建画布，1行3列子图，设置尺寸
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ['Best Case', 'Random Case', 'Worst Case']
        indices = [best_idx, random_idx, worst_idx]
        colors = ['green', 'blue', 'red']

        # 绘制每个子图
        for i, ax in enumerate(axes):
            idx = indices[i]
            ax.plot(final_trues[idx], label='Ground Truth', color='black', linewidth=1.5)  # 真实值
            ax.plot(final_preds[idx], label='MM-LSTM Predict', color=colors[i], linestyle='--', linewidth=1.5)  # 预测值
            ax.set_title(f'{titles[i]} (MAPE: {sample_mapes[idx]:.2f}%)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)  # 网格线增强可读性
        
        plt.tight_layout()  # 自动调整子图间距，防止重叠
        plt.savefig(PLOT_RESULT_PATH, dpi=300)  # 保存高清图片
        logger.info(f"Forecast comparison figure saved to: {PLOT_RESULT_PATH}")

        # 绘图2：训练损失曲线 → 训练集loss vs 验证集loss
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss (MSE)', color='blue', linewidth=1.2)
        plt.plot(history['val_loss'], label='Validation Loss (MSE)', color='orange', linewidth=1.2)
        plt.title('MM-LSTM Training & Validation Loss Curve', fontsize=14)
        plt.xlabel('Training Epoch', fontsize=12)
        plt.ylabel('MSE Loss Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(PLOT_LOSS_PATH, dpi=300)
        logger.info(f"Training loss curve saved to: {PLOT_LOSS_PATH}")
        plt.close('all')  # 关闭所有画布，释放内存

    except Exception as e:
        # 捕获绘图异常，不终止程序，仅记录日志
        logger.error(f"Plotting process failed with error: {str(e)}")
    
    # 实验结束日志
    logger.info(f"\n>>> All Experiment Process Completed Successfully! <<<")
    logger.info(f"All experiment results are saved in directory: {LOG_ROOT}")