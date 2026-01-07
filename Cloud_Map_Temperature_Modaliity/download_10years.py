import cdsapi
import os

# 初始化客户端
c = cdsapi.Client()

# 定义要下载的年份范围 (2008 到 2018)
# range(start, end) 是不包含 end 的，所以写 2019
years = range(2008, 2019) 

# 定义 PJM 区域 (北, 西, 南, 东)
# 涵盖宾夕法尼亚、新泽西等 PJM 东部核心区
AREA_COORDS = [42, -80, 38, -74]

print(f"开始下载 {years[0]} 到 {years[-1]} 年的气象数据...")
print("策略：每年保存为一个独立的 .nc 文件，防止网络中断导致前功尽弃。")

for year in years:
    year_str = str(year)
    output_filename = f'download_weather_{year}.nc'
    
    # 1. 断点续传检查
    if os.path.exists(output_filename):
        print(f"[{year_str}] 文件 {output_filename} 已存在，跳过。")
        continue
        
    print(f"[{year_str}] 正在请求下载... (请耐心等待排队)")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'total_cloud_cover', # 云量 (图像模态)
                    '2m_temperature',    # 温度 (图像/文本模态)
                ],
                'year': year_str,
                'month': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24',
                    '25', '26', '27', '28', '29', '30', '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
                ],
                'area': AREA_COORDS,
            },
            output_filename)
        print(f"[{year_str}] 下载成功！已保存为 {output_filename}")
        
    except Exception as e:
        print(f"[{year_str}] 下载出错: {e}")
        # 这里可以选择 break 停止，或者 continue 尝试下一年
        # 建议出错就停止，让你检查原因
        break

print("所有任务处理完毕。")