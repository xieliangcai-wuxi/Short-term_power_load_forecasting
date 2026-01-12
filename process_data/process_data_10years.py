import pandas as pd
import xarray as xr
import numpy as np
import cv2
import os
import logging

# ==========================================
# 1. 顶刊级日志与规范配置
# ==========================================
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
    IMG_SIZE = (32, 32)
    # 物理约束阈值 (针对 PJME 数据集)
    LOAD_MIN, LOAD_MAX = 10000, 70000 # MW

def run_data_fusion():
    cfg = FusionConfig()

    # --- Step A: 负荷数据清洗与异常检测 ---
    df_load = pd.read_csv(cfg.LOAD_PATH)
    df_load['Datetime'] = pd.to_datetime(df_load['Datetime'])
    df_load = df_load.set_index('Datetime').sort_index()
    df_load = df_load[~df_load.index.duplicated(keep='first')]
    
    # 顶刊严谨性：物理范围检查
    valid_mask = (df_load['PJME_MW'] >= cfg.LOAD_MIN) & (df_load['PJME_MW'] <= cfg.LOAD_MAX)
    if not valid_mask.all():
        logger.warning(f"检测到 {(~valid_mask).sum()} 条异常负荷记录，已执行线性插值填充")
        df_load.loc[~valid_mask, 'PJME_MW'] = np.nan
        df_load['PJME_MW'] = df_load['PJME_MW'].interpolate(method='linear')

    # --- Step B: 语义文本读取 ---
    df_text = pd.read_csv(cfg.TEXT_PATH)
    df_text['datetime'] = pd.to_datetime(df_text['datetime'])
    # 核心：时区对齐（由 UTC 转至当地电力市场时区）
    df_text['datetime'] = df_text['datetime'] + pd.Timedelta(hours=cfg.TIMEZONE_OFFSET)
    df_text = df_text.set_index('datetime').sort_index()

    # --- Step C: 气象图像流式处理 (Memory-Efficient) ---
    logger.info("正在建立气象数据索引...")
    ds_list = []
    for y in cfg.YEARS:
        f_path = cfg.NC_PATTERN.format(y)
        if os.path.exists(f_path):
            # chunks={} 开启 dask 延迟加载，防止 10 年数据撑爆内存
            ds_tmp = xr.open_dataset(f_path, chunks={'time': 500})
            if 'valid_time' in ds_tmp:
                ds_tmp = ds_tmp.rename({'valid_time': 'time'})
            ds_list.append(ds_tmp)
    
    ds_all = xr.concat(ds_list, dim='time')
    ds_all['time'] = ds_all['time'] + pd.Timedelta(hours=cfg.TIMEZONE_OFFSET)
    
    # --- Step D: 三位一体精确时间对齐 ---
    logger.info("执行多模态时间步 Inner Join...")
    # 提取公共时间点
    load_times = df_load.index
    text_times = df_text.index
    nc_times = ds_all.time.to_index()
    
    common_index = load_times.intersection(text_times).intersection(nc_times).sort_values()
    logger.info(f"对齐完成。有效样本数: {len(common_index)}")

    # --- Step E: 图像重采样与归一化 (核心物理特征提取) ---
    final_len = len(common_index)
    final_images = np.zeros((final_len, 2, *cfg.IMG_SIZE), dtype=np.float32)
    
    # 获取对齐后的气象数据
    ds_aligned = ds_all.sel(time=common_index).compute()
    tcc_vals = ds_aligned['tcc'].values # Total Cloud Cover [0, 1]
    t2m_vals = ds_aligned['t2m'].values # Temperature at 2m [K]

    for i in range(final_len):
        # 物理归一化：温度使用常用气象区间 [253.15, 313.15] (即 -20C 到 40C)
        # 这是为了让模型在不同季节的数据输入下保持数值敏感性
        norm_temp = (t2m_vals[i] - 253.15) / (313.15 - 253.15)
        norm_temp = np.clip(norm_temp, 0, 1) # 边界裁剪

        # 使用双线性插值进行重采样
        final_images[i, 0] = cv2.resize(tcc_vals[i], cfg.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        final_images[i, 1] = cv2.resize(norm_temp, cfg.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        
        if i % 10000 == 0:
            logger.info(f"图像处理进度: {i}/{final_len}")

    # --- Step F: 持久化保存 ---
    np.savez_compressed(
        cfg.OUTPUT_PATH,
        load=df_load.loc[common_index, 'PJME_MW'].values.astype(np.float32),
        text=df_text.loc[common_index, 'final_text'].values,
        images=final_images,
        times=common_index.values.astype(str),
        # 记录归一化元数据，这在论文的 Method 章节必须写明
        meta={'temp_min': -20, 'temp_max': 40, 'unit': 'Celsius'}
    )
    logger.info(f"=== 融合成功！数据已存至: {cfg.OUTPUT_PATH} ===")

if __name__ == "__main__":
    try:
        run_data_fusion()
    except Exception as e:
        logger.error(f"流程中断: {str(e)}")