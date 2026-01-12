import pandas as pd
import os
from universal_text_generator import UniversalTextGenerator, DatasetConfig


SETTINGS = {
    "PJM": {
        "country": "US",
        "offset": -5,
        "nc_dir": "../Cloud_Map_Temperature_Modaliity/",
        "prefix": "download_weather_"
    },
    "NEM_NSW": {
        "country": "AU",
        "offset": 10,
        "nc_dir": "../Australian_Weather_Data/",
        "prefix": "aus_nsw_"
    }
}

# 切换此参数生成不同国家的数据集
ACTIVE_TASK = "PJM" 
YEARS = range(2008, 2019)
OUTPUT_FILENAME = f'universal_text_{ACTIVE_TASK}_10years.csv'

# ==========================================
# 批量执行逻辑
# ==========================================
all_dfs = []
task_info = SETTINGS[ACTIVE_TASK]

print(f"=== Starting High-Standard Dataset Generation [{ACTIVE_TASK}] ===")

for year in YEARS:
    nc_file = os.path.join(task_info["nc_dir"], f"{task_info['prefix']}{year}.nc")
    
    if not os.path.exists(nc_file):
        print(f"[-] Skip {year}: File not found.")
        continue
        
    print(f"[*] Processing Year: {year}...")
    
    # 实例化配置与生成器
    cfg = DatasetConfig(
        country_code=task_info["country"],
        timezone_offset=task_info["offset"],
        nc_file_path=nc_file,
        region_name=ACTIVE_TASK
    )
    
    try:
        generator = UniversalTextGenerator(cfg)
        df_year = generator.generate()
        all_dfs.append(df_year[['datetime', 'final_text']])
    except Exception as e:
        print(f"[!] Error in {year}: {e}")

if all_dfs:
    final_df = pd.concat(all_dfs).sort_values('datetime')
    final_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n[SUCCESS] Dataset saved to {OUTPUT_FILENAME}")
    print(f"Total samples: {len(final_df)}")
else:
    print("\n[FAILED] No data generated.")