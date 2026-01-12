import pandas as pd
import xarray as xr
import numpy as np
from workalendar.registry import registry
from datetime import timedelta
import os

class DatasetConfig:
    def __init__(self, country_code: str, timezone_offset: int, nc_file_path: str, region_name: str = "Default"):
        self.country_code = country_code    # 如 'US', 'AU'
        self.timezone_offset = timezone_offset 
        self.nc_file_path = nc_file_path     
        self.region_name = region_name      

class UniversalTextGenerator:
    def __init__(self, config: DatasetConfig):
        self.cfg = config
        # 核心：动态加载全球日历注册表
        try:
            CalendarClass = registry.get(config.country_code)
            self.cal = CalendarClass()
            print(f"[*] Success: Loaded global calendar for {config.country_code}")
        except Exception as e:
            print(f"[!] Warning: Failed to load calendar for {config.country_code}. Fallback to weekend logic.")
            self.cal = None

    def _get_social_modality(self, local_dt):
        """
        社会语义生成逻辑
        """
        day_of_week = local_dt.strftime('%A')
        is_holiday = self.cal.is_holiday(local_dt) if self.cal else False
        
        # 1. 识别具体的节假日名称
        holiday_name = None
        if is_holiday:
            # 这里的 get_holiday_label 是获取具体名称（如 Christmas, Labor Day）
            holidays_list = self.cal.holidays(local_dt.year)
            for h_date, h_name in holidays_list:
                if h_date == local_dt.date():
                    holiday_name = h_name
                    break

        # 2. 构建多层次社会语义
        semantic_parts = [f"Today is {day_of_week}."]
        
        if is_holiday:
            name_str = f" ({holiday_name})" if holiday_name else ""
            semantic_parts.append(f"Observing Public Holiday{name_str}. Industrial activity is significantly curtailed.")
        elif local_dt.weekday() >= 5:
            semantic_parts.append("Weekend status: Commercial clusters active, manufacturing sectors offline.")
        else:
            semantic_parts.append("Standard workday: Expect intensive industrial and office building energy demand.")
            
        return " ".join(semantic_parts)

    def _get_weather_impact(self, t_c):
        """物理-语义映射"""
        if t_c > 32: return "High thermal stress; massive air-conditioning load expected."
        if t_c < 5:  return "Freezing conditions; surge in electric heating requirements."
        return "Moderate weather; baseline temperature-sensitive load."

    def generate(self):
        """
        生成函数：确保时间轴对齐与社会-物理特征融合
        """
        if not os.path.exists(self.cfg.nc_file_path):
            raise FileNotFoundError(f"Missing NC file: {self.cfg.nc_file_path}")

        ds = xr.open_dataset(self.cfg.nc_file_path, engine='netcdf4')
        if 'valid_time' in ds: ds = ds.rename({'valid_time': 'time'})
        
        ds_mean = ds.mean(dim=['latitude', 'longitude'])
        utc_times = pd.to_datetime(ds_mean.time.values)
        temps_k = ds_mean['t2m'].values

        results = []
        for i, utc_time in enumerate(utc_times):
            # 严格时区转换：确保“节假日”是当地的节假日
            local_dt = utc_time + timedelta(hours=self.cfg.timezone_offset)
            t_c = temps_k[i] - 273.15

            # A. 社会语义 
            social_text = self._get_social_modality(local_dt)
            
            # B. 气象语义
            weather_text = self._get_weather_impact(t_c)
            
            # C. 时间步语义 (时刻点对用电习惯的影响)
            hour = local_dt.hour
            time_slot = "Nighttime; residential base load."
            if 7 <= hour < 9: time_slot = "Morning ramp-up period."
            elif 9 <= hour < 18: time_slot = "Peak business and industrial session."
            elif 18 <= hour < 22: time_slot = "Residential peak; evening social activities."

            final_text = f"{social_text} {time_slot} Context: {weather_text}"
            
            results.append({
                'datetime': utc_time, # 用于索引对齐
                'final_text': final_text
            })
            
        return pd.DataFrame(results)