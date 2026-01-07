import pandas as pd
import os
from universal_gen import UniversalTextGenerator, RegionConfig

# ================= 配置 =================
YEARS = range(2008, 2019) # 2008-2018
OUTPUT_FILE = 'universal_text_10years.csv'

all_dfs = []

print(f"=== 开始生成 10 年语义文本 ({YEARS[0]}-{YEARS[-1]}) ===")

for year in YEARS:
    nc_file = f'../Cloud_Map_Temperature_Modaliity/download_weather_{year}.nc'
    
    if not os.path.exists(nc_file):
        print(f"警告: 文件 {nc_file} 不存在，跳过该年份。")
        continue
        
    print(f"\n>> 处理年份: {year}")
    
    # 配置生成器 (PJM 区域)
    config = RegionConfig(
        country_code='US', 
        region_code='PA', 
        timezone_offset=-5, # 美东时间
        nc_file_path=nc_file
    )
    
    try:
        # 实例化并生成
        gen = UniversalTextGenerator(config)
        df_year = gen.generate()
        
        # 只保留需要的列
        df_year = df_year[['datetime', 'final_text']]
        all_dfs.append(df_year)
        print(f"   {year}年生成完成，样本数: {len(df_year)}")
        
    except Exception as e:
        print(f"   {year}年处理出错: {e}")

# 合并所有年份
if all_dfs:
    print("\n正在合并所有年份数据...")
    df_final = pd.concat(all_dfs, ignore_index=True)
    
    # 按时间排序，防止乱序
    df_final = df_final.sort_values('datetime')
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"全部完成！已保存为 {OUTPUT_FILE}")
    print(f"总样本数: {len(df_final)}")
else:
    print("错误: 没有生成任何数据。")