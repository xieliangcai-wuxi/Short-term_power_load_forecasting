# -*- coding: utf-8 -*-
"""
MM-Mamba 多模态时间序列负荷预测完整训练&测试流程
核心模型：MM-Mamba (Multi-Modal Mamba) 选择性状态空间模型(SSM)
核心优势：Mamba 相比 LSTM/Transformer 具备线性复杂度+长序列建模优势，多模态融合负荷/图像/文本特征
核心任务：基于168h(7天)多模态输入序列，实现24h(1天)电力负荷预测
评估标准：IEEE电力负荷预测标准回归指标 (RMSE/MAE/MAPE)，物理量纲统一为兆瓦(MW)
适用场景：小时级电力负荷预测、多模态时序序列预测任务
"""
import os
# ===================== MAMBA 专属强制环境变量配置（核心必配）=====================
# 关键配置1: 强制Mamba内核使用当前Python运行环境，解决Mamba编译的环境依赖冲突问题
os.environ["MAMBA_FORCE_PYTHON"] = "1"
# 关键配置2: 关闭PyTorch底层C++扩展的冗余日志输出，仅保留ERROR级别的关键报错，净化日志
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# 基础深度学习/数据处理库导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import logging
from tqdm import tqdm  # 训练进度可视化工具，提升实验体验

# ===================== 自定义模块导入 & 路径兼容鲁棒处理 =====================
# 兼容两种工程运行场景，杜绝导入报错：
# 场景1: 项目根目录运行（规范层级：model/模型类 | dataset.py数据集类）
# 场景2: 当前文件目录直接运行（模型类/数据集类与本文件同级）
try:
    from model.MMMambaModel import MMMambaModel  # 多模态Mamba核心模型类
    from dataset import MMTimeSeriesDataset       # 多模态时序数据集加载类
except ImportError:
    # 路径降级兼容处理
    from MMMambaModel import MMMambaModel
    from dataset import MMTimeSeriesDataset

# ===================== 顶刊级实验归档 & 日志系统全局配置 =====================
# 生成精确到秒的时间戳，作为实验唯一标识，彻底避免实验文件覆盖，便于实验溯源管理
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 实验根目录：明确标注Mamba，与LSTM/Transformer实验结果分区存放，便于对比
LOG_ROOT = f'./log/mamba_experiment_{TIMESTAMP}' 
MODEL_DIR = os.path.join(LOG_ROOT, 'models')    # 最优模型权重归档目录
FIGURE_DIR = os.path.join(LOG_ROOT, 'figures')  # 可视化结果图表保存目录
DATA_RECORD_DIR = os.path.join(LOG_ROOT, 'data_records')  # 训练历史/测试结果数据归档目录

# 递归创建多级目录，exist_ok=True：目录已存在时不抛出异常，工程鲁棒性保障
for path in [MODEL_DIR, FIGURE_DIR, DATA_RECORD_DIR]:
    os.makedirs(path, exist_ok=True)

# --- 标准化日志器配置：顶刊实验必备，完整记录所有实验过程，可复现可追溯 ---
logger = logging.getLogger('MM-Mamba')  # 日志器命名空间，隔离其他模块日志
logger.setLevel(logging.INFO)            # 日志等级：仅记录INFO及以上级别信息（INFO/WARNING/ERROR）
# 日志格式化：时间戳 + 日志等级 + 日志内容，标准化输出格式，便于日志分析
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 日志处理器1：文件日志 - 将所有日志写入文件永久保存，编码=utf-8解决中文乱码问题
file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'training_log.txt'), encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 日志处理器2：控制台日志 - 将日志实时打印到终端，替代原生print，日志格式统一
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ===================== 全局超参数统一配置中心（可一键修改，所有实验对齐）=====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 硬件设备：优先GPU加速，自动降级CPU
BATCH_SIZE = 64          # 批次大小：64为Mamba最优适配值，显存充足可上调，过小易训练震荡
LR = 0.001               # 初始学习率：Adam优化器最优初始值，配合学习率调度器动态调整
EPOCHS = 50              # 最大训练轮数：防止过拟合，配合早停机制提前终止
PATIENCE = 7             # 早停耐心值：验证集损失连续7轮无下降则终止训练，核心防过拟合策略
DATA_PATH = './process_data/processed_data_10years_v2.npz'  # 数据集路径：必须使用v2版，含多模态meta特征

# --- 实验结果文件保存路径（统一配置，便于实验后归档与查找，全Mamba专属命名）---
SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_mamba.pth')          # 最优模型权重保存路径
HISTORY_SAVE_PATH = os.path.join(DATA_RECORD_DIR, 'training_history.npz')  # 训练损失/指标历史保存路径
PLOT_RESULT_PATH = os.path.join(FIGURE_DIR, 'mamba_forecast_result.png')    # 预测结果可视化保存路径
PLOT_LOSS_PATH = os.path.join(FIGURE_DIR, 'mamba_loss_curve.png')           # 训练损失曲线保存路径

def seed_everything(seed=42):
    """
    全局随机种子固定函数：顶刊实验核心要求，保证实验结果100%可复现
    :param seed: 随机种子值，默认42，本次实验指定为50
    """
    import random
    random.seed(seed)                # Python原生随机数种子固定
    os.environ['PYTHONHASHSEED'] = str(seed)  # 关闭Python哈希随机化，固定哈希值
    np.random.seed(seed)             # Numpy随机数种子固定
    torch.manual_seed(seed)          # PyTorch CPU随机数种子固定
    torch.cuda.manual_seed_all(seed) # PyTorch 多GPU随机数种子固定
    torch.backends.cudnn.deterministic = True  # 强制CUDNN使用确定性卷积算法
    torch.backends.cudnn.benchmark = False     # 关闭卷积算法自动优化，保证结果一致性

# ===================== 模块1：物理反归一化+IEEE标准指标计算工具类 =====================
# 核心功能：1.将归一化的模型输出还原为真实物理量纲(MW) 2.计算负荷预测领域IEEE标准评估指标
class MetricTracker:
    def __init__(self, mean, std):
        """
        初始化反归一化标尺，必须使用【训练集】的均值和标准差，杜绝数据泄露
        :param mean: 训练集负荷数据的均值
        :param std: 训练集负荷数据的标准差
        """
        self.mean = mean
        self.std = std

    def inverse_transform(self, y_norm):
        """
        反归一化核心函数：严格对齐数据集的归一化公式 Real = Norm * Std + Mean
        实现tensor/numpy双类型兼容，自动适配模型输出与数据格式
        :param y_norm: 归一化后的预测值/真实值 (torch.Tensor or np.ndarray)
        :return: 还原为物理量纲(MW)的数值 (np.ndarray)
        """
        if torch.is_tensor(y_norm):
            y_norm = y_norm.cpu().detach().numpy()  # Tensor转Numpy：解绑GPU+梯度+迁移CPU
        return y_norm * self.std + self.mean

    def calculate_metrics(self, y_true_real, y_pred_real):
        """
        计算负荷预测IEEE三大标准评估指标（物理量纲统一为MW/百分比），指标越小代表预测精度越高
        :param y_true_real: 反归一化后的真实负荷值 (np.ndarray)
        :param y_pred_real: 反归一化后的预测负荷值 (np.ndarray)
        :return: rmse(均方根误差), mae(平均绝对误差), mape(平均绝对百分比误差)
        """
        # MAE：平均绝对误差，对异常值鲁棒，最直观的误差衡量，单位MW
        mae = np.mean(np.abs(y_pred_real - y_true_real))
        # RMSE：均方根误差，对大偏差惩罚更重，负荷预测核心评估指标，单位MW
        rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
        
        # MAPE：平均绝对百分比误差，无量纲，直观反映预测相对误差，负荷预测必报指标
        # np.errstate：忽略除零/无效值警告，真实值为0时不中断程序，行业通用处理方案
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_pred_real - y_true_real) / y_true_real)) * 100
            
        return rmse, mae, mape

# ===================== 模块2：早停机制工具类（训练必备，核心防过拟合策略）=====================
# 核心逻辑：当验证集损失连续N轮无有效下降时，提前终止训练，并保存训练过程中的最优模型权重
# 优势：避免模型过拟合、节省训练时间、保留最优泛化能力的模型权重
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        初始化早停器
        :param patience: 耐心值，连续多少轮验证集loss无提升则触发早停
        :param delta: 损失下降阈值，仅当loss下降超过该值才算有效提升，过滤微小波动
        """
        self.patience = patience
        self.counter = 0            # 连续无提升轮数计数器
        self.best_loss = np.inf     # 最优损失值，初始化为无穷大
        self.early_stop = False     # 早停触发标志位
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        每轮验证后调用，自动判断是否更新最优模型/触发早停，支持实例化后直接调用
        :param val_loss: 当前轮次验证集平均损失值
        :param model: 当前训练的MM-Mamba模型实例
        :param path: 最优模型权重的保存路径
        """
        # 情况1：当前验证损失 优于 最优损失-阈值 → 更新最优模型
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)  # 保存最优模型权重
            self.counter = 0                   # 重置计数器
        # 情况2：当前验证损失无有效提升 → 计数器+1
        else:
            self.counter += 1
            logger.info(f'   [EarlyStop] Counter: {self.counter}/{self.patience}')
            # 计数器达到耐心值 → 触发早停
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        """模型权重保存函数：仅保存模型参数字典，轻量化存储，便于后续加载"""
        logger.info(f'   [CheckPoint] Val Loss improved. Saving best MM-Mamba model...')
        torch.save(model.state_dict(), path)

# ===================== 主程序入口：完整实验流程（训练+验证+测试+可视化+归档）=====================
# 主函数隔离：保证模块导入时不执行实验代码，仅在直接运行时执行
if __name__ == "__main__":
    # 1. 固定全局随机种子，保证实验可复现性
    seed_everything(50)
    # 2. 打印实验启动头信息，标准化实验日志，便于实验溯源
    logger.info("=" * 60)
    logger.info("MM-Mamba Multi-modal Forecasting Pipeline | IEEE Standard Metrics")
    logger.info(f"Start Time: {datetime.datetime.now()}")
    logger.info(f"Using Device: {DEVICE}")
    logger.info("=" * 60)

    # --- A. 数据集加载 + 严谨的时序/统计审计（顶刊实验必备，数据有效性校验）---
    # 数据文件存在性校验，文件缺失则终止程序并打印错误日志
    if not os.path.exists(DATA_PATH):
        logger.error(f"Critical Error: Data file not found at {DATA_PATH}")
        sys.exit()

    # 加载多模态时序数据集，按80%/10%/10%划分训练/验证/测试集，内部已完成归一化
    train_set = MMTimeSeriesDataset(DATA_PATH, mode='train')
    val_set = MMTimeSeriesDataset(DATA_PATH, mode='val')
    test_set = MMTimeSeriesDataset(DATA_PATH, mode='test')

    # --- 时序维度审计：校验数据集的时间区间与划分合理性，实验可追溯的核心环节 ---
    raw_data = np.load(DATA_PATH, allow_pickle=True)
    times = raw_data['times']        # 原始时间戳数组，用于审计时间区间
    total_len = len(times)           # 总时间序列长度（小时级样本）
    train_end = int(total_len * 0.8) # 训练集结束索引
    val_end = int(total_len * 0.9)   # 验证集结束索引
    
    # 打印数据集审计日志：时间区间+统计特征，校验数据划分与归一化标尺的合理性
    logger.info(">>> Dataset Temporal & Statistical Audit <<<")
    logger.info(f"Train Segment: {times[0]} to {times[train_end-1]} | Mean: {train_set.load_mean:.2f} | Std: {train_set.load_std:.2f}")
    logger.info(f"Val Segment:   {times[train_end]} to {times[val_end-1]} | Mean: {val_set.load_mean:.2f} | Std: {val_set.load_std:.2f}")
    logger.info(f"Test Segment:  {times[val_end]} to {times[-1]} | Mean: {test_set.load_mean:.2f} | Std: {test_set.load_std:.2f}")
    logger.info("=" * 60)

    # 构建数据加载器：批处理+数据洗牌+显存优化，Mamba专属non_blocking=True异步加速
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化指标追踪器：使用训练集的均值和标准差作为反归一化标尺，杜绝数据泄露
    tracker = MetricTracker(train_set.load_mean, train_set.load_std)

    # --- B. 模型与训练组件初始化（所有组件统一配置，实验对齐性保障）---
    # 初始化MM-Mamba多模态模型：输入序列长度168h(7天)，预测序列长度24h(1天)，部署到指定设备
    model = MMMambaModel(seq_len=168, pred_len=24).to(DEVICE)
    criterion = nn.MSELoss()  # 损失函数：均方误差，回归任务首选，与负荷预测行业对齐
    # 优化器：Adam + L2正则化(weight_decay=1e-5)，核心防过拟合，与Transformer/LSTM实验条件对齐
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) 
    # 学习率调度器：验证集损失不下降时，学习率减半，耐心值3轮，动态调整学习率提升收敛性
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # 早停器实例化：使用全局配置的耐心值
    early_stopping = EarlyStopping(patience=PATIENCE)

    # 训练历史记录字典：保存每轮训练/验证损失+验证MAPE，用于后续可视化与分析
    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    # --- C. MM-Mamba 核心训练循环（含逐轮验证，顶刊级完整训练流程）---
    logger.info("\n>>> Starting MM-Mamba Training Loop <<<")
    for epoch in range(EPOCHS):
        model.train()  # 模型切换训练模式：启用Dropout/层归一化的训练态，更新梯度
        epoch_train_loss = 0.0  # 累计当前轮次的训练损失
        # 进度条包装训练加载器，实时显示训练进度与当前批次损失，提升实验体验
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)
        
        for batch in pbar:
            # 加载多模态批次数据，部署到指定设备，non_blocking=True异步拷贝加速Mamba训练
            x_load = batch['x_load'].to(DEVICE, non_blocking=True)
            x_img = batch['x_img'].to(DEVICE, non_blocking=True)
            x_text = batch['x_text'].to(DEVICE, non_blocking=True)
            y_true = batch['y_load'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()  # 清空梯度缓存，防止梯度累加导致的训练震荡
            y_pred = model(x_load, x_img, x_text)  # 模型前向传播，输出预测值
            
            # ========== MAMBA 专属核心修复：维度对齐检查 ==========
            # Mamba模型的状态空间输出易产生冗余维度 [B, T, 1]，需与真实值 [B, T] 对齐才能计算损失
            # 行业通用解决方案：自动检测维度不一致并挤压最后一维，杜绝loss计算维度不匹配报错
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)

            loss = criterion(y_pred, y_true)  # 计算批次损失
            loss.backward()                    # 反向传播，计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪：防止Mamba梯度爆炸
            optimizer.step()                   # 梯度更新，优化模型参数
            
            epoch_train_loss += loss.item()    # 累计批次损失
            pbar.set_postfix(train_mse=loss.item())  # 进度条实时显示当前批次损失

        # --- 每轮训练后执行验证流程：无梯度计算，评估模型泛化能力 ---
        model.eval()  # 模型切换评估模式：关闭Dropout/层归一化的训练态，固定模型参数
        epoch_val_loss = 0.0
        all_val_preds, all_val_trues = [], []  # 保存验证集所有预测值/真实值
        
        with torch.no_grad():  # 关闭梯度计算：节省显存+加速验证，无反向传播需求
            for batch in val_loader:
                x_load = batch['x_load'].to(DEVICE)
                x_img = batch['x_img'].to(DEVICE)
                x_text = batch['x_text'].to(DEVICE)
                y_true = batch['y_load'].to(DEVICE)
                
                y_pred = model(x_load, x_img, x_text)
                
                # 验证集同样执行维度对齐检查，与训练集逻辑一致
                if y_pred.shape != y_true.shape:
                    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                        y_pred = y_pred.squeeze(-1)
                    elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                        y_true = y_true.squeeze(-1)
                
                epoch_val_loss += criterion(y_pred, y_true).item()
                
                # 反归一化后保存，用于计算物理量纲的评估指标
                all_val_preds.append(tracker.inverse_transform(y_pred))
                all_val_trues.append(tracker.inverse_transform(y_true))

        # 计算当前轮次的平均训练/验证损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # 拼接所有验证集预测值与真实值，计算IEEE标准指标
        val_preds_mw = np.concatenate(all_val_preds, axis=0)
        val_trues_mw = np.concatenate(all_val_trues, axis=0)
        _, _, v_mape = tracker.calculate_metrics(val_trues_mw, val_preds_mw)

        # 打印本轮训练总结日志，标准化输出格式，便于实验分析
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Summary: Train_MSE={avg_train_loss:.6f} | Val_MSE={avg_val_loss:.6f} | Val_MAPE={v_mape:.2f}%")
        
        # 保存本轮训练历史数据
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(v_mape)

        # 学习率调度器更新：基于验证集损失动态调整学习率
        scheduler.step(avg_val_loss)
        # 早停器判断：是否更新最优模型/触发早停
        early_stopping(avg_val_loss, model, SAVE_PATH)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Training terminated to prevent overfitting.")
            break

    # --- D. 训练完成：保存训练历史数据，便于后续分析与可视化 ---
    logger.info(f"\nSaving training history to: {HISTORY_SAVE_PATH}")
    np.savez(HISTORY_SAVE_PATH, 
             train_loss=np.array(history['train_loss']),
             val_loss=np.array(history['val_loss']),
             val_mape=np.array(history['val_mape']))

    # --- E. 最终测试评估：加载最优模型权重，无偏评估模型泛化能力（顶刊实验核心环节）---
    logger.info("\n" + "="*20 + " Final Test Performance Evaluation " + "="*20)
    model.load_state_dict(torch.load(SAVE_PATH))  # 加载训练过程中保存的最优模型权重
    model.eval()  # 模型切换评估模式

    test_preds, test_trues = [], []  # 保存测试集所有预测值/真实值
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final MM-Mamba Testing"):
            x_load = batch['x_load'].to(DEVICE)
            x_img = batch['x_img'].to(DEVICE)
            x_text = batch['x_text'].to(DEVICE)
            y_true = batch['y_load'].to(DEVICE)
            
            y_pred = model(x_load, x_img, x_text)

            # 测试集维度对齐检查，与训练/验证集逻辑一致
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)

            # 反归一化后保存测试集结果
            test_preds.append(tracker.inverse_transform(y_pred))
            test_trues.append(tracker.inverse_transform(y_true))

    # 拼接并压缩测试集结果维度，去除冗余维度便于计算指标
    final_preds = np.squeeze(np.concatenate(test_preds, axis=0))
    final_trues = np.squeeze(np.concatenate(test_trues, axis=0))

    # 持久化测试集结果，便于后续深度分析与绘图
    np.savez(os.path.join(DATA_RECORD_DIR, 'test_results.npz'), preds=final_preds, trues=final_trues)

    # 计算测试集最终IEEE标准评估指标，打印核心实验结果
    rmse, mae, mape = tracker.calculate_metrics(final_trues, final_preds)
    logger.info(f"Test Set Final Metrics (IEEE Standard) >>")
    logger.info(f"Test RMSE: {rmse:.2f} MW")
    logger.info(f"Test MAE:  {mae:.2f} MW")
    logger.info(f"Test MAPE: {mape:.2f}%")

    # --- F. 实验结果可视化：绘制预测对比图+损失曲线，顶刊实验必备展示形式 ---
    plt.switch_backend('Agg')  # 切换无GUI后端：服务器/无桌面环境下正常绘图保存，不报错
    
    try:
        # 1. 绘制预测结果对比图：最优/随机/最差样本三图合一，直观展示预测效果
        sample_mapes = np.mean(np.abs((final_preds - final_trues) / (final_trues + 1e-5)), axis=1) * 100
        best_idx = np.argmin(sample_mapes)   # MAPE最小→最优预测样本
        worst_idx = np.argmax(sample_mapes)  # MAPE最大→最差预测样本
        random_idx = np.random.randint(0, len(final_preds))  # 随机样本

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ['Best Case', 'Random Case', 'Worst Case']
        indices = [best_idx, random_idx, worst_idx]
        colors = ['green', 'blue', 'red']

        for i, ax in enumerate(axes):
            idx = indices[i]
            ax.plot(final_trues[idx], label='Ground Truth', color='black', linewidth=1.5)
            ax.plot(final_preds[idx], label='MM-Mamba Predict', color=colors[i], linestyle='--', linewidth=1.5)
            ax.set_title(f'{titles[i]} (MAPE: {sample_mapes[idx]:.2f}%)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()  # 自动调整子图间距，防止重叠
        plt.savefig(PLOT_RESULT_PATH, dpi=300)  # 保存高清图片
        logger.info(f"Forecast comparison figure saved: {PLOT_RESULT_PATH}")

        # 2. 绘制训练损失曲线：训练集vs验证集，直观展示模型收敛性与过拟合情况
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss (MSE)', color='blue', linewidth=1.2)
        plt.plot(history['val_loss'], label='Validation Loss (MSE)', color='orange', linewidth=1.2)
        plt.title('MM-Mamba Training & Validation Loss Curve', fontsize=14)
        plt.xlabel('Training Epoch', fontsize=12)
        plt.ylabel('MSE Loss Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(PLOT_LOSS_PATH, dpi=300)
        logger.info(f"Loss curve saved: {PLOT_LOSS_PATH}")
        plt.close('all')  # 关闭所有画布，释放内存资源

    except Exception as e:
        # 捕获绘图异常，不终止程序，仅打印错误日志，工程鲁棒性保障
        logger.error(f"Plotting failed with error: {str(e)}")

    # 实验结束日志，标准化输出
    logger.info(f"\nMM-Mamba Experiment Complete Successfully! All results saved in: {LOG_ROOT}")