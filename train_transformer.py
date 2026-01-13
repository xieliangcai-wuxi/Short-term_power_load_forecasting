# -*- coding: utf-8 -*-
"""
MM-Transformer 多模态时间序列电力负荷预测完整训练&测试流程
核心模型：MM-Transformer (Multi-Modal Transformer) 多头自注意力融合模型
核心优势：Transformer 自注意力机制可建模长时序依赖+多模态特征间的关联关系
核心任务：基于168h(7天)负荷/图像/文本多模态输入序列，实现24h(1天)小时级电力负荷预测
评估标准：IEEE电力负荷预测领域标准回归指标 (RMSE/MAE/MAPE)，物理量纲统一为兆瓦(MW)
核心适配：针对Transformer易过拟合/梯度爆炸特性做全维度工程优化，保证训练稳定性
"""
import os
import sys
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 训练进度可视化工具，提升实验过程可读性

# ===================== 自定义模块导入 & 鲁棒性路径兼容处理 =====================
# 兼容两种工业界标准工程运行场景，彻底杜绝导入报错，无需手动修改路径：
# 场景1: 项目根目录运行（规范层级结构：model/模型类 | dataset.py数据集类）
# 场景2: 当前文件目录直接运行（模型类/数据集类与本文件同级目录）
try:
    from model.MMTransformerModel import MMTransformerModel  # 多模态Transformer核心模型类
    from dataset import MMTimeSeriesDataset                   # 多模态时序数据集加载类
except ImportError:
    # 路径降级兼容处理：捕获导入异常后使用同级目录导入
    from MMTransformerModel import MMTransformerModel
    from dataset import MMTimeSeriesDataset

# ===================== 顶刊级实验归档 & 日志系统全局标准化配置 =====================
# 生成精确到秒的时间戳，作为实验唯一标识，彻底避免实验文件覆盖，实现实验全溯源管理
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 实验根目录：明确标注Transformer，与LSTM/Mamba实验结果分区存放，便于多模型对比分析
LOG_ROOT = f'./log/transformer_experiment_{TIMESTAMP}' 

# 实验子目录划分：功能解耦，归档规范，顶刊实验必备的工程化素养
MODEL_DIR = os.path.join(LOG_ROOT, 'models')        # 最优模型权重归档目录
FIGURE_DIR = os.path.join(LOG_ROOT, 'figures')      # 可视化结果图表保存目录
DATA_RECORD_DIR = os.path.join(LOG_ROOT, 'data_records')  # 训练历史/测试结果数据归档目录

# 递归创建多级目录，exist_ok=True：目录已存在时不抛出异常，保障代码鲁棒性
for path in [MODEL_DIR, FIGURE_DIR, DATA_RECORD_DIR]:
    os.makedirs(path, exist_ok=True)

# --- 标准化日志器配置（替代原生print，顶刊实验必备）---
logger = logging.getLogger('MM-Transformer')  # 日志命名空间隔离，避免与其他模块日志混淆
logger.setLevel(logging.INFO)                  # 日志等级：仅记录 INFO/WARNING/ERROR 级信息，净化日志输出
# 日志格式化：时间戳 + 日志等级 + 日志内容，标准化输出格式，便于后续日志分析与实验复盘
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 日志处理器1：文件日志 - 永久保存所有实验过程日志，编码=utf-8彻底解决中文乱码问题
file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'training_log.txt'), encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 日志处理器2：控制台日志 - 实时打印日志到终端，与文件日志格式统一，便于实时监控训练状态
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ===================== 全局超参数统一配置中心（一键修改，所有实验对齐，无硬编码）=====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 硬件设备：自动适配GPU/CPU，优先GPU加速
BATCH_SIZE = 64          # 批次大小：64为Transformer最优适配值，兼顾训练效率与稳定性，显存充足可上调
LR = 0.001               # 初始学习率：Adam优化器最优初始值，配合学习率调度器动态调整
EPOCHS = 50              # 最大训练轮数：Transformer易过拟合，配合早停机制提前终止训练
PATIENCE = 7             # 早停耐心值：验证集损失连续7轮无下降则终止训练，核心防过拟合策略
DATA_PATH = './process_data/processed_data_10years_v2.npz'  # 数据集路径【关键修正】：必须使用v2版，含多模态meta特征

# --- 实验结果文件保存路径【关键修复】补全所有缺失路径定义，统一管理便于修改 ---
SAVE_PATH = os.path.join(MODEL_DIR, 'best_mm_transformer.pth')          # 最优模型权重保存路径
HISTORY_SAVE_PATH = os.path.join(DATA_RECORD_DIR, 'training_history.npz')  # 训练损失/指标历史保存路径
PLOT_RESULT_PATH = os.path.join(FIGURE_DIR, 'transformer_forecast_result.png')    # 预测结果可视化路径
PLOT_LOSS_PATH = os.path.join(FIGURE_DIR, 'transformer_loss_curve.png')           # 训练损失曲线保存路径

def seed_everything(seed=42):
    """
    全局随机种子固定函数：顶刊实验核心硬性要求，保证实验结果100%可复现，无随机误差
    Transformer对随机种子极其敏感，固定种子是多模型对比实验的前提
    :param seed: 随机种子值，本次实验指定50，与LSTM/Mamba基线模型完全对齐
    """
    import random
    random.seed(seed)                # Python原生随机数种子固定
    os.environ['PYTHONHASHSEED'] = str(seed)  # 关闭Python哈希随机化，固定哈希值
    np.random.seed(seed)             # Numpy随机数种子固定
    torch.manual_seed(seed)          # PyTorch CPU随机数种子固定
    torch.cuda.manual_seed_all(seed) # PyTorch 多GPU随机数种子固定
    torch.backends.cudnn.deterministic = True  # 强制CUDNN使用确定性卷积算法
    torch.backends.cudnn.benchmark = False     # 关闭卷积算法自动优化，保证结果一致性

# ===================== 模块1：物理反归一化 + IEEE标准指标计算工具类 =====================
# 核心功能：1.将归一化的模型输出还原为真实物理量纲(MW) 2.计算负荷预测领域IEEE三大标准评估指标
# 核心原则：使用【训练集】的均值/标准差作为反归一化标尺，彻底杜绝数据泄露，符合顶刊实验规范
class MetricTracker:
    def __init__(self, mean, std):
        """
        初始化反归一化标尺
        :param mean: 训练集负荷数据的均值
        :param std: 训练集负荷数据的标准差
        """
        self.mean = mean
        self.std = std

    def inverse_transform(self, y_norm):
        """
        反归一化核心函数：严格对齐数据集的归一化公式 → Real = Norm * Std + Mean
        自动兼容Tensor/Numpy两种数据类型，适配模型输出与指标计算的不同格式需求
        :param y_norm: 归一化后的预测值/真实值 (torch.Tensor or np.ndarray)
        :return: 还原为物理量纲(MW)的数值 (np.ndarray)
        """
        if torch.is_tensor(y_norm):
            y_norm = y_norm.cpu().detach().numpy()  # Tensor转Numpy：解绑GPU+梯度+迁移至CPU，无内存泄漏
        return y_norm * self.std + self.mean

    def calculate_metrics(self, y_true_real, y_pred_real):
        """
        计算IEEE电力负荷预测领域三大标准评估指标，指标越小代表预测精度越高，物理意义明确
        :param y_true_real: 反归一化后的真实负荷值 (np.ndarray)
        :param y_pred_real: 反归一化后的预测负荷值 (np.ndarray)
        :return: rmse(均方根误差), mae(平均绝对误差), mape(平均绝对百分比误差)
        """
        # MAE：平均绝对误差，对异常值鲁棒，最直观的误差衡量，单位 MW
        mae = np.mean(np.abs(y_pred_real - y_true_real))
        # RMSE：均方根误差，对大偏差惩罚更重，负荷预测核心评估指标，单位 MW
        rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
        
        # MAPE：平均绝对百分比误差，无量纲，直观反映预测相对误差，顶刊必报指标
        # 【关键修正】移除+1.0偏置，使用np.errstate忽略除零/无效值警告，行业通用最优解决方案
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_pred_real - y_true_real) / y_true_real)) * 100
        return rmse, mae, mape

# ===================== 模块2：早停机制工具类（Transformer训练必备，核心防过拟合策略）=====================
# 核心逻辑：当验证集损失连续N轮无有效下降时，提前终止训练，并保存训练过程中的最优模型权重
# Transformer天生具有极强的拟合能力，极易发生过拟合，早停是必备的正则化手段
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        初始化早停器
        :param patience: 耐心值，连续多少轮验证集loss无提升则触发早停
        :param delta: 损失下降阈值，仅当loss下降超过该值才算有效提升，过滤微小波动，避免频繁保存模型
        """
        self.patience = patience
        self.counter = 0            # 连续无提升轮数计数器
        self.best_loss = np.inf     # 最优损失值，初始化为无穷大
        self.early_stop = False     # 早停触发标志位
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        每轮验证后自动调用，判断是否更新最优模型/触发早停，支持实例化后直接调用
        :param val_loss: 当前轮次验证集平均损失值
        :param model: 当前训练的MM-Transformer模型实例
        :param path: 最优模型权重的保存路径
        """
        # 情况1：当前验证损失 优于 最优损失-阈值 → 更新最优模型权重
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
            self.counter = 0
        # 情况2：当前验证损失无有效提升 → 计数器+1
        else:
            self.counter += 1
            logger.info(f'   [EarlyStop] Counter: {self.counter}/{self.patience}')
            # 计数器达到耐心值 → 触发早停，终止训练防止过拟合
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        """模型权重轻量化保存：仅保存参数字典，不保存模型结构，节省存储空间"""
        logger.info(f'   [CheckPoint] Val Loss improved. Saving best MM-Transformer model...')
        torch.save(model.state_dict(), path)

# ===================== 主程序入口：完整实验流程（训练+验证+测试+可视化+归档）=====================
# 主函数隔离：保证模块导入时不执行实验代码，仅在直接运行时执行，符合Python工程规范
if __name__ == "__main__":
    # 1. 固定全局随机种子【关键】：与LSTM/Mamba基线模型使用相同种子，保证对比实验公平性
    seed_everything(50) 

    # 2. 打印实验启动标准化头信息，便于实验溯源与复盘
    logger.info("=" * 60)
    logger.info("MM-Transformer Multi-modal Forecasting Pipeline | IEEE Standard Metrics")
    logger.info(f"Start Time: {datetime.datetime.now()}")
    logger.info(f"Using Device: {DEVICE}")
    logger.info("=" * 60)

    # --- A. 数据集加载 + 严谨的时序/统计审计（顶刊实验必备，数据有效性校验，杜绝脏数据）---
    # 数据文件存在性校验，文件缺失则终止程序并打印错误日志，避免程序崩溃
    if not os.path.exists(DATA_PATH):
        logger.error(f"Critical Error: Data file not found at {DATA_PATH}")
        sys.exit()
    
    # 实例化多模态时序数据集，按80%/10%/10%严格划分训练/验证/测试集，内部已完成归一化预处理
    train_set = MMTimeSeriesDataset(DATA_PATH, mode='train')
    val_set = MMTimeSeriesDataset(DATA_PATH, mode='val')
    test_set = MMTimeSeriesDataset(DATA_PATH, mode='test')

    # --- 时序维度审计 (Temporal Audit)：顶刊实验必做环节，校验数据集划分合理性与完整性 ---
    raw_data = np.load(DATA_PATH, allow_pickle=True)
    times = raw_data['times']        # 原始时间戳数组，用于审计数据的时间区间
    total_len = len(times)           # 总时间序列长度（小时级样本）
    train_end = int(total_len * 0.8) # 训练集结束索引
    val_end = int(total_len * 0.9)   # 验证集结束索引
    
    # 打印数据集审计日志：时间区间+统计特征，校验数据划分与归一化标尺的合理性，实验可追溯
    logger.info(">>> Dataset Temporal & Statistical Audit <<<")
    logger.info(f"Train Segment: {times[0]} to {times[train_end-1]} | Mean: {train_set.load_mean:.2f} | Std: {train_set.load_std:.2f}")
    logger.info(f"Val Segment:   {times[train_end]} to {times[val_end-1]} | Mean: {val_set.load_mean:.2f} | Std: {val_set.load_std:.2f}")
    logger.info(f"Test Segment:  {times[val_end]} to {times[-1]} | Mean: {test_set.load_mean:.2f} | Std: {test_set.load_std:.2f}")
    logger.info("=" * 60)

    # 构建数据加载器：批处理+数据洗牌+显存优化，针对Transformer做专属加速配置
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化指标追踪器：使用训练集的均值和标准差作为反归一化标尺，彻底杜绝数据泄露
    tracker = MetricTracker(train_set.load_mean, train_set.load_std)

    # --- B. 模型与训练组件初始化（Transformer专属配置，全组件对齐基线模型，实验公平性保障）---
    try:
        # 初始化MM-Transformer多模态模型，核心参数显性配置，便于调优与复现
        # seq_len=168(输入7天序列), pred_len=24(预测1天序列), d_model=128(特征维度), n_layers=2(编码器层数), nhead=4(多头注意力数)
        # 核心约束：d_model必须是nhead的整数倍，否则多头注意力机制会报错
        model = MMTransformerModel(seq_len=168, pred_len=24, d_model=128, n_layers=2, nhead=4).to(DEVICE)
    except AttributeError:
        # 【关键兼容修复】捕获类导入路径异常，兼容模型类嵌套定义的情况，彻底解决模型实例化报错
        model = MMTransformerModel.MMTransformerModel(seq_len=168, pred_len=24, d_model=128, n_layers=2, nhead=4).to(DEVICE)
        
    criterion = nn.MSELoss()  # 损失函数：均方误差，回归任务首选，与LSTM/Mamba基线模型完全对齐
    # 优化器：Adam + L2正则化(weight_decay=1e-5)【关键配置】，Transformer必加的防过拟合手段
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    # 学习率调度器：验证集损失不下降时学习率减半，耐心值3轮，动态调整学习率提升模型收敛性
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # 早停器实例化：使用全局配置的耐心值，核心防过拟合策略
    early_stopping = EarlyStopping(patience=PATIENCE)

    # 训练历史记录字典：保存每轮训练/验证损失+验证MAPE，用于后续可视化与实验分析
    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

    # --- C. MM-Transformer 核心训练循环（含逐轮验证，全流程工程化优化）---
    logger.info("\n>>> Starting Training Loop (MM-Transformer) <<<")
    for epoch in range(EPOCHS):
        model.train()  # 模型切换训练模式：启用Dropout/层归一化的训练态，允许梯度更新
        epoch_train_loss = 0.0  # 累计当前轮次的训练损失
        # 进度条包装训练加载器，实时显示训练进度与当前批次损失，提升实验体验
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', file=sys.stdout)
        
        for batch in pbar:
            # 加载多模态批次数据，部署到指定设备，non_blocking=True异步拷贝加速，Transformer显存优化关键
            x_load = batch['x_load'].to(DEVICE, non_blocking=True)
            x_img = batch['x_img'].to(DEVICE, non_blocking=True)
            x_text = batch['x_text'].to(DEVICE, non_blocking=True)
            y_true = batch['y_load'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()  # 清空梯度缓存，防止梯度累加导致的训练震荡
            y_pred = model(x_load, x_img, x_text)  # 模型前向传播，输出预测值
            
            # --- Transformer 专属维度防御检查【核心修复】---
            # Transformer的线性层输出易产生冗余维度 [B, T, 1]，需与真实值 [B, T] 对齐才能计算损失
            # 自动检测维度不一致并挤压最后一维，杜绝loss计算维度不匹配报错，保障训练稳定性
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)

            loss = criterion(y_pred, y_true)  # 计算批次损失
            loss.backward()                    # 反向传播，计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪【Transformer必备】：防止自注意力梯度爆炸
            optimizer.step()                   # 梯度更新，优化模型参数
            
            epoch_train_loss += loss.item()    # 累计批次损失
            pbar.set_postfix(train_mse=loss.item())  # 进度条实时显示当前批次损失

        # --- 每轮训练后执行验证流程：无梯度计算，公平评估模型泛化能力 ---
        model.eval()  # 模型切换评估模式：关闭Dropout/层归一化的训练态，固定模型参数，无梯度更新
        epoch_val_loss = 0.0
        all_val_preds, all_val_trues = [], []  # 保存验证集所有预测值/真实值
        
        with torch.no_grad():  # 关闭梯度计算：节省显存+加速验证，验证阶段无反向传播需求
            for batch in val_loader:
                x_load = batch['x_load'].to(DEVICE)
                x_img = batch['x_img'].to(DEVICE)
                x_text = batch['x_text'].to(DEVICE)
                y_true = batch['y_load'].to(DEVICE)
                
                y_pred = model(x_load, x_img, x_text)
                
                # 验证集维度对齐检查，与训练集逻辑一致，保障损失计算正确性
                if y_pred.shape != y_true.shape:
                    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                        y_pred = y_pred.squeeze(-1)
                    elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                        y_true = y_true.squeeze(-1)

                epoch_val_loss += criterion(y_pred, y_true).item()
                
                # 反归一化后保存，用于计算物理量纲的IEEE标准评估指标
                all_val_preds.append(tracker.inverse_transform(y_pred))
                all_val_trues.append(tracker.inverse_transform(y_true))
        
        # 计算当前轮次的平均训练/验证损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # 拼接所有验证集预测值与真实值，计算IEEE标准评估指标
        val_preds_mw = np.concatenate(all_val_preds, axis=0)
        val_trues_mw = np.concatenate(all_val_trues, axis=0)
        _, _, v_mape = tracker.calculate_metrics(val_trues_mw, val_preds_mw)
        
        # 打印本轮训练总结日志，标准化输出格式，便于实验分析与多轮对比
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val MAPE: {v_mape:.2f}%")
        
        # 保存本轮训练历史数据，用于后续可视化与实验复盘
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(v_mape)

        # 学习率调度器更新：基于验证集损失动态调整学习率，提升模型收敛性
        scheduler.step(avg_val_loss)
        # 早停器判断：是否更新最优模型/触发早停
        early_stopping(avg_val_loss, model, SAVE_PATH)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Training terminated to prevent overfitting.")
            break

    # --- D. 训练完成：保存训练历史数据，便于后续分析与可视化，实验结果可复现 ---
    logger.info(f"\nSaving training history to: {HISTORY_SAVE_PATH}")
    np.savez(HISTORY_SAVE_PATH, 
             train_loss=np.array(history['train_loss']),
             val_loss=np.array(history['val_loss']),
             val_mape=np.array(history['val_mape']))

    # --- E. 最终测试评估【顶刊实验核心环节】：加载最优模型权重，无偏评估模型泛化能力 ---
    logger.info("\n" + "="*20 + " Final Test Performance Evaluation " + "="*20)
    model.load_state_dict(torch.load(SAVE_PATH))  # 加载训练过程中保存的最优模型权重，保证测试公平性
    model.eval()  # 模型切换评估模式

    test_preds, test_trues = [], []  # 保存测试集所有预测值/真实值
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="MM-Transformer Final Testing", file=sys.stdout):
            x_load = batch['x_load'].to(DEVICE)
            x_img = batch['x_img'].to(DEVICE)
            x_text = batch['x_text'].to(DEVICE)
            y_true = batch['y_load'].to(DEVICE)
            
            y_pred = model(x_load, x_img, x_text)
            
            # 测试集维度对齐检查，与训练/验证集逻辑一致，保障指标计算正确性
            if y_pred.shape != y_true.shape:
                if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)
                elif y_true.dim() == 3 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)

            # 反归一化后保存测试集结果，用于计算最终评估指标
            test_preds.append(tracker.inverse_transform(y_pred))
            test_trues.append(tracker.inverse_transform(y_true))

    # 拼接并压缩测试集结果维度，去除冗余维度，便于指标计算与可视化
    final_preds = np.squeeze(np.concatenate(test_preds, axis=0))
    final_trues = np.squeeze(np.concatenate(test_trues, axis=0))

    # 持久化测试集结果，便于后续深度分析与绘图，实验结果可追溯
    np.savez(os.path.join(DATA_RECORD_DIR, 'test_results.npz'), preds=final_preds, trues=final_trues)

    # 计算测试集最终IEEE标准评估指标，打印核心实验结果，顶刊论文核心数据来源
    rmse, mae, mape = tracker.calculate_metrics(final_trues, final_preds)
    logger.info(f"Test RMSE: {rmse:.2f} MW")
    logger.info(f"Test MAE:  {mae:.2f} MW")
    logger.info(f"Test MAPE: {mape:.2f}%")

    # --- F. 实验结果可视化：顶刊实验必备展示形式，直观展示预测效果与训练收敛性 ---
    plt.switch_backend('Agg')  # 切换无GUI后端：服务器/无桌面环境下正常绘图保存，不报错，工程鲁棒性保障
    
    try:
        # 1. 绘制预测结果对比图：最优/随机/最差样本三图合一，直观展示模型预测效果的优劣分布
        sample_mapes = np.mean(np.abs((final_preds - final_trues) / (final_trues + 1e-5)), axis=1) * 100
        best_idx = np.argmin(sample_mapes)   # MAPE最小 → 最优预测样本
        worst_idx = np.argmax(sample_mapes)  # MAPE最大 → 最差预测样本
        random_idx = np.random.randint(0, len(final_preds))  # 随机样本，反映模型平均性能

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ['Best Case', 'Random Case', 'Worst Case']
        indices = [best_idx, random_idx, worst_idx]
        colors = ['green', 'blue', 'red']

        for i, ax in enumerate(axes):
            idx = indices[i]
            ax.plot(final_trues[idx], label='Ground Truth', color='black', linewidth=1.5)
            ax.plot(final_preds[idx], label='MM-Transformer', color=colors[i], linestyle='--', linewidth=1.5)
            ax.set_title(f'{titles[i]} (MAPE: {sample_mapes[idx]:.2f}%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()  # 自动调整子图间距，防止重叠，提升可视化美观度
        plt.savefig(PLOT_RESULT_PATH, dpi=300)  # 保存高清图片，满足顶刊投稿要求
        logger.info(f"Forecast comparison figure saved: {PLOT_RESULT_PATH}")

        # 2. 绘制训练损失曲线：训练集vs验证集，直观展示模型收敛性与过拟合情况
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['val_loss'], label='Val Loss', color='orange')
        plt.title('MM-Transformer Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(PLOT_LOSS_PATH, dpi=300)
        logger.info(f"Loss curve saved: {PLOT_LOSS_PATH}")
        plt.close('all')  # 关闭所有画布，释放内存资源，无内存泄漏

    except Exception as e:
        # 捕获绘图异常，不终止程序，仅打印错误日志，工程鲁棒性保障
        logger.error(f"Plotting failed with error: {str(e)}")

    # 实验结束标准化日志，便于实验溯源与归档
    logger.info(f"\nMM-Transformer Experiment Complete Successfully! All results saved in: {LOG_ROOT}")