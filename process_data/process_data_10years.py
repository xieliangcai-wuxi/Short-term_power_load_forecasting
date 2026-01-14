import pandas as pd
import xarray as xr
import numpy as np
import cv2
import os
import logging
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FusionConfig:
    """配置类"""
    LOAD_PATH = '../Historical_Load_Data_Modality/PJME_hourly.csv'
    TEXT_PATH = '../Social_Semantic_Modality/universal_text_PJM_10years.csv'
    NC_PATTERN = '../Cloud_Map_Temperature_Modaliity/download_weather_{}.nc'
    OUTPUT_PATH = './processed_data_10years_v2.npz'
    
    YEARS = range(2008, 2019)
    TIMEZONE_OFFSET = -5
    IMG_SIZE = (64, 64)
    # 物理约束阈值 (针对 PJME 数据集)
    LOAD_MIN, LOAD_MAX = 10000, 70000 # MW
    # 划分比例 - 用于提取归一化标尺
    TRAIN_SPLIT_RATIO = 0.8 

def run_data_fusion():
    cfg = FusionConfig()
    logger.info("=== 启动多模态数据融合流程 ===")

    # --- Step A: 负荷数据清洗与异常检测 ---
    df_load = pd.read_csv(cfg.LOAD_PATH)
    df_load['Datetime'] = pd.to_datetime(df_load['Datetime'])
    df_load = df_load.set_index('Datetime').sort_index()
    df_load = df_load[~df_load.index.duplicated(keep='first')]
    
    # 物理范围检查与插值修复
    valid_mask = (df_load['PJME_MW'] >= cfg.LOAD_MIN) & (df_load['PJME_MW'] <= cfg.LOAD_MAX)
    if not valid_mask.all():
        logger.warning(f"检测到 {(~valid_mask).sum()} 条异常负荷记录，已执行线性插值填充")
        df_load.loc[~valid_mask, 'PJME_MW'] = np.nan
        df_load['PJME_MW'] = df_load['PJME_MW'].interpolate(method='linear')

    # --- Step B: 语义文本读取 ---
    df_text = pd.read_csv(cfg.TEXT_PATH)
    df_text['datetime'] = pd.to_datetime(df_text['datetime'])
    # 时区对齐 (UTC -> Local)
    df_text['datetime'] = df_text['datetime'] + pd.Timedelta(hours=cfg.TIMEZONE_OFFSET)
    df_text = df_text.set_index('datetime').sort_index()

    # --- Step C: 气象图像流式处理 (Memory-Efficient) ---
    logger.info("正在建立气象数据索引并加载 NC 文件...")
    ds_list = []
    for y in cfg.YEARS:
        f_path = cfg.NC_PATTERN.format(y)
        if os.path.exists(f_path):
            ds_tmp = xr.open_dataset(f_path, chunks={'time': 500}) # 启用 dask
            if 'valid_time' in ds_tmp:
                ds_tmp = ds_tmp.rename({'valid_time': 'time'})
            ds_list.append(ds_tmp)
    
    ds_all = xr.concat(ds_list, dim='time')
    ds_all['time'] = ds_all['time'] + pd.Timedelta(hours=cfg.TIMEZONE_OFFSET)
    
    # --- Step D: 三位一体精确时间对齐 ---
    logger.info("执行多模态时间步对齐 (Inner Join)...")
    load_times = df_load.index
    text_times = df_text.index
    nc_times = ds_all.time.to_index()
    
    common_index = load_times.intersection(text_times).intersection(nc_times).sort_values()
    logger.info(f"对齐完成。总有效样本数: {len(common_index)}")

    # --- Step E: 提取训练集统计量 ---
    # 锁定前 TRAIN_SPLIT_RATIO 的数据作为计算均值和标准差的基准
    split_idx = int(len(common_index) * cfg.TRAIN_SPLIT_RATIO)
    train_index = common_index[:split_idx]
    train_load_values = df_load.loc[train_index, 'PJME_MW'].values
    
    train_mean = float(train_load_values.mean())
    train_std = float(train_load_values.std())
    logger.info(f"训练集统计特征已锁定: Mean={train_mean:.2f}, Std={train_std:.2f}")

    # --- Step F: 图像重采样与物理归一化 ---
    final_len = len(common_index)
    final_images = np.zeros((final_len, 2, *cfg.IMG_SIZE), dtype=np.float32)
    
    # compute() 此时将对齐后的图像载入内存
    ds_aligned = ds_all.sel(time=common_index).compute()
    tcc_vals = ds_aligned['tcc'].values
    t2m_vals = ds_aligned['t2m'].values 

    for i in range(final_len):
        norm_temp = (t2m_vals[i] - 253.15) / (313.15 - 253.15)
        norm_temp = np.clip(norm_temp, 0, 1)

        # 使用双线性插值进行重采样 (自动适配 64x64)
        final_images[i, 0] = cv2.resize(tcc_vals[i], cfg.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        final_images[i, 1] = cv2.resize(norm_temp, cfg.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        
        if i % 10000 == 0:
            logger.info(f"图像处理进度: {i}/{final_len}")

    # 这一步是为了防止梯度消失，确保存入的数据包含有效信息
    logger.info(">>> Image Channel Physical Audit (Before Saving) <<<")
    
    # Channel 0: Cloud Cover
    cloud_mean = final_images[:, 0, :, :].mean()
    cloud_std = final_images[:, 0, :, :].std()
    cloud_max = final_images[:, 0, :, :].max()
    logger.info(f"Channel 0 [Cloud]: Mean={cloud_mean:.4f}, Std={cloud_std:.4f}, Max={cloud_max:.4f} (Expected: ~0.5, >0.1, 1.0)")
    
    # Channel 1: Temperature
    temp_mean = final_images[:, 1, :, :].mean()
    temp_std = final_images[:, 1, :, :].std()
    logger.info(f"Channel 1 [Temp] : Mean={temp_mean:.4f}, Std={temp_std:.4f} (Expected: ~0.5, >0.1)")
    
    if cloud_std < 1e-3 or temp_std < 1e-3:
        logger.warning("!!! 警告: 图像通道方差过低，可能导致模型无法学习，请检查原始 NC 文件 !!!")
    else:
        logger.info(">>> Audit Passed: Image features are statistically significant. <<<")
    
    # --- Step G: 持久化保存 (含训练集元数据) ---
    final_load = df_load.loc[common_index, 'PJME_MW'].values.astype(np.float32)
    final_text = df_text.loc[common_index, 'final_text'].values
    final_times_str = common_index.values.astype(str)

    np.savez_compressed(
        cfg.OUTPUT_PATH,
        load=final_load,
        text=final_text,
        images=final_images,
        times=final_times_str,
        meta={
            'train_mean': train_mean, 
            'train_std': train_std,
            'temp_min': -20, 
            'temp_max': 40,
            'unit': 'MW',
            'split_ratio': cfg.TRAIN_SPLIT_RATIO
        }
    )
    logger.info(f"=== 融合成功！数据集已存至: {cfg.OUTPUT_PATH} ===")

if __name__ == "__main__":
    try:
        run_data_fusion()
    except Exception as e:
        logger.error(f"融合流程出错: {str(e)}")