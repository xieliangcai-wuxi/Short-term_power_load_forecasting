import pandas as pd
import xarray as xr
import numpy as np
import cv2
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FusionConfig:
    # 请确认路径正确
    LOAD_PATH = '../Historical_Load_Data_Modality/PJME_hourly.csv'
    TEXT_PATH = '../Social_Semantic_Modality/universal_text_PJM_10years.csv' 
    NC_PATTERN = '../Cloud_Map_Temperature_Modaliity/download_weather_{}.nc'
    OUTPUT_PATH = './processed_data_final.npz' # 生成新文件，不覆盖旧的
    
    YEARS = range(2008, 2019)
    IMG_SIZE = (64, 64)
    LOAD_MIN, LOAD_MAX = 10000, 70000 
    TRAIN_SPLIT_RATIO = 0.8 

def run_data_fusion():
    cfg = FusionConfig()
    logger.info("=== 启动数据融合 (DST 修复版) ===")

    # --- Step A: 负荷数据清洗 (带夏令时处理) ---
    logger.info("正在加载负荷数据并修复时区...")
    df_load = pd.read_csv(cfg.LOAD_PATH)
    df_load['Datetime'] = pd.to_datetime(df_load['Datetime'])
    
    # 1. 去重和排序
    df_load = df_load.sort_values('Datetime').set_index('Datetime')
    # 处理夏令时回拨导致的重复时间 (02:00 出现两次)，保留第一次出现的
    df_load = df_load[~df_load.index.duplicated(keep='first')]
    
    # 2. 赋予时区属性 (US/Eastern)
    # ambiguous='infer' 尝试推断，nonexistent='shift_forward' 跳过春季消失的那一小时
    try:
        df_load = df_load.tz_localize('US/Eastern', ambiguous='infer', nonexistent='shift_forward')
    except Exception:
        # 如果 infer 失败，强制转换
        df_load = df_load.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='shift_forward')
        df_load = df_load.dropna()

    # 3. 转为 UTC (这是关键！与气象对齐)
    df_load = df_load.tz_convert('UTC')
    
    # 物理清洗
    df_load.loc[(df_load['PJME_MW'] < cfg.LOAD_MIN) | (df_load['PJME_MW'] > cfg.LOAD_MAX), 'PJME_MW'] = np.nan
    df_load['PJME_MW'] = df_load['PJME_MW'].interpolate(method='linear')

    # --- Step B: 语义文本读取 ---
    # 假设你新生成的文本 csv 已经是 UTC 或者包含时区信息
    # 如果是用我刚才给的 UniversalTextGenerator 生成的，它已经是 UTC 了
    df_text = pd.read_csv(cfg.TEXT_PATH)
    df_text['datetime'] = pd.to_datetime(df_text['datetime'])
    if df_text['datetime'].dt.tz is None:
        df_text['datetime'] = df_text['datetime'].dt.tz_localize('UTC')
    else:
        df_text['datetime'] = df_text['datetime'].dt.tz_convert('UTC')
    df_text = df_text.set_index('datetime').sort_index()

    # --- Step C: 气象图像加载 ---
    logger.info("加载气象 NC 文件...")
    ds_list = []
    for y in cfg.YEARS:
        f_path = cfg.NC_PATTERN.format(y)
        if os.path.exists(f_path):
            ds_tmp = xr.open_dataset(f_path)
            if 'valid_time' in ds_tmp: ds_tmp = ds_tmp.rename({'valid_time': 'time'})
            ds_list.append(ds_tmp)
    
    ds_all = xr.concat(ds_list, dim='time')
    # ERA5 默认就是 UTC，只需确保是 Pandas Timestamp 格式
    weather_times = pd.to_datetime(ds_all.time.values)
    if weather_times.tz is None:
        weather_times = weather_times.tz_localize('UTC')
    ds_all = ds_all.assign_coords(time=weather_times)

    # --- Step D: 对齐 (Inner Join on UTC) ---
    logger.info("执行 UTC 对齐...")
    common_index = df_load.index.intersection(df_text.index).intersection(ds_all.time.to_index()).sort_values()
    logger.info(f"对齐样本数: {len(common_index)}")

    # --- Step E: 统计量计算 (只用训练集) ---
    split_idx = int(len(common_index) * cfg.TRAIN_SPLIT_RATIO)
    train_index = common_index[:split_idx]
    train_vals = df_load.loc[train_index, 'PJME_MW'].values
    
    train_mean = float(train_vals.mean())
    train_std = float(train_vals.std())
    logger.info(f"Train Mean: {train_mean:.2f}, Std: {train_std:.2f}")

    # --- Step F: 图像处理 ---
    final_len = len(common_index)
    final_images = np.zeros((final_len, 2, *cfg.IMG_SIZE), dtype=np.float32)
    
    ds_aligned = ds_all.sel(time=common_index).compute()
    tcc = ds_aligned['tcc'].values # Cloud
    t2m = ds_aligned['t2m'].values # Temp (Kelvin)

    for i in range(final_len):
        # Temp norm (K -> 0-1)
        norm_temp = (t2m[i] - 253.15) / (313.15 - 253.15)
        norm_temp = np.clip(norm_temp, 0, 1)
        
        # Resize
        final_images[i, 0] = cv2.resize(tcc[i], cfg.IMG_SIZE)
        final_images[i, 1] = cv2.resize(norm_temp, cfg.IMG_SIZE)
        if i % 10000 == 0: logger.info(f"Processed {i}/{final_len}")

    # --- Step G: 保存 ---
    final_load = df_load.loc[common_index, 'PJME_MW'].values.astype(np.float32)
    # 只要文本列
    final_text = df_text.loc[common_index, 'final_text'].values 
    
    np.savez_compressed(
        cfg.OUTPUT_PATH,
        load=final_load,
        text=final_text,
        images=final_images,
        times=common_index.astype(str), # 保存 UTC 时间字符串
        meta={
            'train_mean': train_mean,
            'train_std': train_std
        }
    )
    logger.info(f"完成！已保存: {cfg.OUTPUT_PATH}")

if __name__ == "__main__":
    run_data_fusion()