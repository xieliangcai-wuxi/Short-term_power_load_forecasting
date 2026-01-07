import pandas as pd
import xarray as xr
import numpy as np
import cv2
import os

# ================= 配置路径 =================
# 请确保这些文件名和你实际的文件名一致
FILE_LOAD = './Historical_Load_Data_Modality/PJME_hourly.csv'
FILE_WEATHER = './Cloud_Map_Temperature_Modaliity/download_weather_2018.nc'
FILE_TEXT = './Social_Semantic_Modality/universal_text_modality_real.csv'
OUTPUT_FILE = './processed_data_final.npz'

TIMEZONE_OFFSET = -5 # PJM (美东) 比 UTC 晚 5 小时

print("=== 开始数据融合流程 ===")

# ================= 1. 读取电力负荷 (Anchor) =================
print(f"1. 读取负荷数据: {FILE_LOAD}")
df_load = pd.read_csv(FILE_LOAD)
df_load['Datetime'] = pd.to_datetime(df_load['Datetime'])
df_load = df_load.set_index('Datetime').sort_index()

# 截取 2018 年的数据 (确保你的 csv 里有这一年的数据)
# PJM 的 csv 可能会有一些重复索引或乱序，清理一下
df_load = df_load[~df_load.index.duplicated(keep='first')]
df_load_2018 = df_load['2018-01-01':'2018-12-31']
print(f"   负荷数据范围: {df_load_2018.index.min()} 到 {df_load_2018.index.max()}")
print(f"   样本数: {len(df_load_2018)}")

# ================= 2. 读取语义文本 (Text) =================
print(f"2. 读取文本数据: {FILE_TEXT}")
df_text = pd.read_csv(FILE_TEXT)
df_text['datetime'] = pd.to_datetime(df_text['datetime'])

# *** 关键：时区对齐 ***
# 文本里的 datetime 是 UTC。我们需要把它变成 Local Time 来匹配负荷。
# 比如 UTC 1月1日 05:00 -> Local 1月1日 00:00
df_text['datetime'] = df_text['datetime'] + pd.Timedelta(hours=TIMEZONE_OFFSET)
df_text = df_text.set_index('datetime').sort_index()

# 截取 2018 (Local Time)
df_text_2018 = df_text['2018-01-01':'2018-12-31']
print(f"   文本数据样本数: {len(df_text_2018)}")

# ================= 3. 读取气象图像 (Image) =================
print(f"3. 读取气象数据: {FILE_WEATHER}")
ds = xr.open_dataset(FILE_WEATHER)

# 自动修复时间变量名 (防止 valid_time 报错)
if 'valid_time' in ds:
    ds = ds.rename({'valid_time': 'time'})

# *** 关键：时区对齐 ***
# 气象数据的 time 也是 UTC，同样平移
ds['time'] = ds['time'] + pd.Timedelta(hours=TIMEZONE_OFFSET)

# 选取 2018 (Local Time)
# 注意：xarray 的切片是包含两端的，可能会多一点点数据，后面会对齐
ds_2018 = ds.sel(time=slice('2018-01-01', '2018-12-31'))
print(f"   气象数据时间点: {len(ds_2018['time'])}")

# ================= 4. 寻找公共交集 (Intersection) =================
print("4. 执行时间对齐 (Intersection)...")

# 找出三个数据源都存在的时间点
common_index = df_load_2018.index.intersection(df_text_2018.index).intersection(ds_2018.time.to_index())
common_index = common_index.sort_values()

print(f"   最终对齐后的样本总数: {len(common_index)} 小时")
if len(common_index) < 100:
    print("错误：对齐后的数据太少！请检查时区设置或年份是否匹配。")
    exit()

# ================= 5. 提取并转换数据 =================
print("5. 正在打包数据 (Resize & Normalize)...")

# A. 提取负荷
# 注意：这里我们保存原始值，归一化放在 Dataset 类里做，方便后面反归一化看结果
final_load = df_load_2018.loc[common_index]['PJME_MW'].values
# 确保是 float32
final_load = final_load.astype(np.float32)

# B. 提取文本
final_text = df_text_2018.loc[common_index]['final_text'].values

# C. 提取图像 (最耗时)
# 预分配数组: [样本数, 2通道, 32高, 32宽]
final_images = np.zeros((len(common_index), 2, 32, 32), dtype=np.float32)

# 提取 xarray 数据
# 确保顺序和 common_index 一致
ds_aligned = ds_2018.sel(time=common_index)
tcc_values = ds_aligned['tcc'].values # 云量
t2m_values = ds_aligned['t2m'].values # 温度

for i in range(len(common_index)):
    if i % 1000 == 0:
        print(f"   处理图像进度: {i}/{len(common_index)}")
    
    # 1. 获取原始图
    raw_cloud = tcc_values[i]
    raw_temp = t2m_values[i]
    
    # 2. 归一化温度 (假设范围 -20C 到 40C, 即 253K 到 313K)
    # 云量本来就是 0-1，不用动
    norm_temp = (raw_temp - 253.0) / (313.0 - 253.0)
    
    # 3. Resize 到 32x32
    # cv2.resize 接收 (width, height)
    img_cloud = cv2.resize(raw_cloud, (32, 32))
    img_temp = cv2.resize(norm_temp, (32, 32))
    
    # 4. 填入
    final_images[i, 0, :, :] = img_cloud
    final_images[i, 1, :, :] = img_temp

# ================= 6. 保存 =================
print(f"6. 保存至 {OUTPUT_FILE} ...")
np.savez(OUTPUT_FILE, 
         load=final_load, 
         images=final_images, 
         text=final_text,
         times=common_index) # 把时间也存下来，画图用

print("=== 数据准备全部完成！ ===")
print(f"Load shape: {final_load.shape}")
print(f"Images shape: {final_images.shape}")
print(f"Text shape: {final_text.shape}")