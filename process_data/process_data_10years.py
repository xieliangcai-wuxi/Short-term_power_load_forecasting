import pandas as pd
import xarray as xr
import numpy as np
import cv2
import os

# ================= 配置 =================
FILE_LOAD = '../Historical_Load_Data_Modality/PJME_hourly.csv'
FILE_TEXT = '../Social_Semantic_Modality/universal_text_10years.csv'
NC_FILES_PATTERN = '../Cloud_Map_Temperature_Modaliity/download_weather_{}.nc' # 文件名模板
YEARS = range(2008, 2019)
OUTPUT_FILE = 'processed_data_10years.npz'
TIMEZONE_OFFSET = -5

print("=== 开始 10 年数据超级融合 ===")

# 1. 读取负荷数据 (Anchor)
print("1. 读取并清洗负荷数据...")
df_load = pd.read_csv(FILE_LOAD)
df_load['Datetime'] = pd.to_datetime(df_load['Datetime'])
df_load = df_load.set_index('Datetime').sort_index()
df_load = df_load[~df_load.index.duplicated(keep='first')] # 去重
# 截取 10 年
df_load = df_load['2008-01-01':'2018-12-31'] 
print(f"   负荷样本数: {len(df_load)}")

# 2. 读取文本数据
print("2. 读取文本数据...")
df_text = pd.read_csv(FILE_TEXT)
df_text['datetime'] = pd.to_datetime(df_text['datetime'])
# 修正时区 (UTC -> Local)
df_text['datetime'] = df_text['datetime'] + pd.Timedelta(hours=TIMEZONE_OFFSET)
df_text = df_text.set_index('datetime').sort_index()
# 截取 10 年
df_text = df_text['2008-01-01':'2018-12-31']
print(f"   文本样本数: {len(df_text)}")

# 3. 批量读取气象数据 (使用 xarray 自动合并)
print("3. 批量读取气象 .nc 文件 (这可能需要一些内存)...")
ds_list = []
for year in YEARS:
    f_path = NC_FILES_PATTERN.format(year)
    if os.path.exists(f_path):
        ds_tmp = xr.open_dataset(f_path)
        # 修正变量名 valid_time -> time
        if 'valid_time' in ds_tmp:
            ds_tmp = ds_tmp.rename({'valid_time': 'time'})
        ds_list.append(ds_tmp)
    else:
        print(f"   警告: 缺少年份 {year} 的文件")

# 合并为一个大的 dataset
ds_all = xr.concat(ds_list, dim='time')
# 修正时区
ds_all['time'] = ds_all['time'] + pd.Timedelta(hours=TIMEZONE_OFFSET)
# 排序
ds_all = ds_all.sortby('time')
# 截取
ds_all = ds_all.sel(time=slice('2008-01-01', '2018-12-31'))
print(f"   气象总时间点: {len(ds_all['time'])}")

# 4. 寻找交集
print("4. 执行时间对齐...")
common_index = df_load.index.intersection(df_text.index).intersection(ds_all.time.to_index())
common_index = common_index.sort_values()
print(f"   最终对齐样本数: {len(common_index)} (预期约 96,000)")

# 5. 转换数据
print("5. 打包 Numpy 数组 (Resize & Save)...")
final_load = df_load.loc[common_index]['PJME_MW'].values.astype(np.float32)
final_text = df_text.loc[common_index]['final_text'].values

# 预分配图像数组
final_images = np.zeros((len(common_index), 2, 32, 32), dtype=np.float32)

# 为了加速，一次性读出 numpy array 再处理，避免循环里频繁 IO
ds_aligned = ds_all.sel(time=common_index)
tcc_vals = ds_aligned['tcc'].values # 云
t2m_vals = ds_aligned['t2m'].values # 温度

# 循环 Resize
total = len(common_index)
for i in range(total):
    if i % 10000 == 0:
        print(f"   进度: {i}/{total}")
        
    raw_cloud = tcc_vals[i]
    raw_temp = t2m_vals[i]
    
    # 归一化温度 (-20C ~ 40C) -> (253K ~ 313K)
    norm_temp = (raw_temp - 253.0) / (313.0 - 253.0)
    
    final_images[i, 0] = cv2.resize(raw_cloud, (32, 32))
    final_images[i, 1] = cv2.resize(norm_temp, (32, 32))

# 6. 保存
np.savez(OUTPUT_FILE, 
         load=final_load, 
         images=final_images, 
         text=final_text,
         times=common_index)
print(f"全部完成！文件保存为: {OUTPUT_FILE}")