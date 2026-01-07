import pandas as pd
import xarray as xr
import numpy as np
from workalendar.registry import registry # 关键：引入动态注册表
import datetime

# ==========================================
# 1. 全球通用配置类 (Configuration)
# ==========================================
class RegionConfig:
    def __init__(self, country_code, region_code, timezone_offset, nc_file_path):
        """
        country_code: 'US', 'CN', 'DE', etc. (ISO 2位代码)
        region_code: 具体州/省代码 (如 'PA' for Pennsylvania), 如果没有则填 None
        timezone_offset: 与 UTC 的时差 (如 PJM东部是 -5, 北京是 +8)
        nc_file_path: 你的 ERA5 .nc 文件路径
        """
        self.country_code = country_code
        self.region_code = region_code
        self.offset = timezone_offset
        self.nc_path = nc_file_path

# ==========================================
# 2. 通用生成器核心类 (Universal Generator)
# ==========================================
class UniversalTextGenerator:
    def __init__(self, config):
        self.cfg = config
        
        # A. 动态加载日历 (无需修改代码，支持全球)
        try:
            CalendarClass = registry.get(config.country_code)
            self.cal = CalendarClass()
            print(f"已加载日历: {config.country_code}")
        except:
            print(f"警告: 未找到 {config.country_code} 的日历，将仅使用周末逻辑。")
            self.cal = None

    def _get_hourly_activity(self, local_dt, is_holiday):
        """通用的人类活动逻辑 (根据当地时间)"""
        h = local_dt.hour
        wd = local_dt.weekday()
        is_weekend = (wd >= 5)
        
        if 0 <= h < 6:
            return "Deep night, minimal activity."
        elif 6 <= h < 8:
            return "Morning commute starts."
        elif 8 <= h < 18:
            if is_holiday:
                return "Public Holiday. Factories closed."
            elif is_weekend:
                return "Weekend. Commercial activity low."
            else:
                return "Working hours. High industrial load."
        elif 18 <= h < 22:
            return "Evening peak. Residential activity high."
        else:
            return "Late night. Load decreasing."

    def _get_weather_impact(self, temp_c):
        """基于真实温度的通用物理影响判断"""
        # 这些阈值是基于人类生理学的，全球通用
        if temp_c > 35:
            return "Extreme heat wave. AC usage max."
        elif temp_c > 30:
            return "Hot weather. High cooling demand."
        elif temp_c > 20:
            return "Comfortable temperature."
        elif temp_c > 10:
            return "Cool weather."
        elif temp_c > 0:
            return "Cold. Heating required."
        else:
            return "Freezing cold. High heating demand."

# ==========================================
# 修改后的 generate 函数 (更健壮)
# ==========================================
    def generate(self):
        print(f"正在读取真实气象数据: {self.cfg.nc_path} ...")
        
        # A. 读取 ERA5 数据
        ds = xr.open_dataset(self.cfg.nc_path)
        
        # --- [关键修改] 自动修复时间变量名 ---
        # 检查是否存在 valid_time，如果存在，把它改名为 time
        if 'valid_time' in ds:
            print("检测到时间变量名为 'valid_time'，正在重命名为 'time'...")
            ds = ds.rename({'valid_time': 'time'})
        elif 'time' not in ds:
            # 如果既没有 time 也没有 valid_time，打印出来让用户看
            print("错误：未找到时间维度！当前所有变量名如下：")
            print(ds.coords)
            raise ValueError("无法找到时间维度 (time 或 valid_time)")
        # ----------------------------------
        
        # B. 空间聚合 (Spatial Aggregation)
        # ERA5 是网格数据(lat, lon)，我们需要把它变成一个区域的平均值
        # 这样就得到了一条单纯的时间曲线
        ds_mean = ds.mean(dim=['latitude', 'longitude'])
        
        # C. 提取时间序列
        # 注意：ERA5 的时间永远是 UTC
        utc_times = pd.to_datetime(ds_mean.time.values)
        temps_k = ds_mean['t2m'].values # 开尔文温度
        
        # D. 生成文本
        results = []
        
        for i, utc_time in enumerate(utc_times):
            # 1. 转换为当地时间 (Local Time)
            # 这是关键！如果不转时区，你的“白天”可能是当地的“半夜”
            local_time = utc_time + datetime.timedelta(hours=self.cfg.offset)
            
            # 2. 获取真实温度 (K -> C)
            real_temp_c = temps_k[i] - 273.15
            
            # 3. 判断日历状态 (使用当地时间)
            is_holiday = False
            if self.cal:
                # workalendar 需要 datetime.date 对象
                is_holiday = self.cal.is_holiday(local_time)
                
            # 4. 生成各部分描述
            # [日期]
            date_desc = f"It is {local_time.day_name()}."
            if is_holiday:
                # 尝试获取节日名称
                try:
                    hol_name = self.cal.get_holiday_label(local_time)
                    date_desc += f" Holiday: {hol_name}."
                except:
                    date_desc += " Public Holiday."
            
            # [活动]
            act_desc = self._get_hourly_activity(local_time, is_holiday)
            
            # [真实气象影响]
            wea_desc = self._get_weather_impact(real_temp_c)
            
            # 5. 组合
            final_text = f"{date_desc} {act_desc} Real Weather Context: {wea_desc}"
            
            # 保存 UTC 时间作为索引 (方便后续和负荷数据对齐)
            results.append({
                'datetime': utc_time, # 保持 UTC 以便与负荷对齐
                'local_time': local_time, # 仅供检查用
                'final_text': final_text
            })
            
        return pd.DataFrame(results)

# ==========================================
# 3. 用户执行区域 (User Execution)
# ==========================================
if __name__ == "__main__":
    
    # --- 场景 A: 你的当前任务 (美国 PJM 地区) ---
    # 步骤：
    # 1. 指向你之前下载的 .nc 文件
    # 2. 设置 US 代码，时区 -5 (EST)
    
    config_pjm = RegionConfig(
        country_code='US',        # 美国
        region_code='PA',         # 宾夕法尼亚 (可选，workalendar支持的话更准)
        timezone_offset=-5,       # 美东时间
        nc_file_path='download_weather_2018.nc' # 必须是你真实存在的文件名
    )
    
    # 实例化并运行
    generator = UniversalTextGenerator(config_pjm)
    
    # 只有当文件存在时才运行，否则报错
    import os
    if os.path.exists(config_pjm.nc_path):
        df_text = generator.generate()
        
        # 保存
        output_file = 'universal_text_modality_real.csv'
        # 只保留 datetime 和 final_text，符合 dataset.py 的要求
        df_text[['datetime', 'final_text']].to_csv(output_file, index=False)
        
        print("-" * 30)
        print(f"生成成功！已保存为 {output_file}")
        print("前 3 条样本 (UTC时间):")
        print(df_text.head(3))
        
        print("\n随机抽取一条检查 (查看真实温度带来的描述):")
        print(df_text.sample(1).iloc[0]['final_text'])
    else:
        print(f"错误：找不到文件 {config_pjm.nc_path}。请确保你运行了之前的下载脚本。")

    # --- 场景 B: 如果你要换成中国 (泛用性演示) ---
    # config_cn = RegionConfig('CN', None, 8, 'china_weather.nc')
    # gen_cn = UniversalTextGenerator(config_cn)
    # ...