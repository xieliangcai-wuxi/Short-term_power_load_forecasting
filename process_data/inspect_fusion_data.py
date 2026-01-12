import numpy as np
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = './processed_data_10years_v2.npz'

def inspect_multimodal_data():
    logger.info(f"正在读取多模态融合数据: {DATA_PATH}")
    
    # 加载数据
    with np.load(DATA_PATH, allow_pickle=True) as data:
        load = data['load']
        images = data['images']
        text = data['text']
        times = data['times']
        # 获取元数据（如果保存时包含字典，读取需 .item()）
        meta = data['meta'].item() if 'meta' in data.files else "No metadata"

    # --- Step A: 基础维度与统计检查 ---
    logger.info("="*30)
    logger.info("基础维度检查 (Shape Check):")
    logger.info(f"  时间步总数 (T): {len(times)}")
    logger.info(f"  负荷数组维度: {load.shape} (Expected: T,)")
    logger.info(f"  气象图像维度: {images.shape} (Expected: T, 2, 32, 32)")
    logger.info(f"  语义文本维度: {text.shape} (Expected: T,)")
    logger.info(f"  元数据信息: {meta}")
    
    # 检查是否存在 NaN 
    if np.isnan(load).any() or np.isnan(images).any():
        logger.error("检测到数据中包含 NaN 值，请重新检查预处理流程！")
    else:
        logger.info("数据完整性检查通过: 无缺失值 (NaN-free).")

    # --- Step B: 随机样本语义对齐检查 ---
    # 抽取一个样本，验证“负荷-图像-文本”在物理逻辑上是否自洽
    idx = np.random.randint(0, len(times))
    logger.info("="*30)
    logger.info(f"样本逻辑对齐体检 (Index: {idx}):")
    logger.info(f"  时间戳 (UTC): {times[idx]}")
    logger.info(f"  负荷数值: {load[idx]:.2f} MW")
    logger.info(f"  语义文本: {text[idx]}")

    # --- Step C: 物理特征可视化 ---
    # 这是最直观的验证方法：看云图和温度图是否对应文本描述
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 负荷全景图 (验证时间轴连续性)
    axes[0].plot(load[:1000], color='blue', linewidth=1)
    axes[0].set_title("Load Profile (First 1000h)")
    axes[0].set_ylabel("MW")
    
    # 2. 当前样本云量图 (Channel 0)
    cloud_map = axes[1].imshow(images[idx, 0], cmap='Blues')
    axes[1].set_title(f"Cloud Cover (tcc)\nSample {idx}")
    fig.colorbar(cloud_map, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. 当前样本温度图 (Channel 1)
    temp_map = axes[2].imshow(images[idx, 1], cmap='RdYlBu_r')
    axes[2].set_title(f"Normalized Temp (t2m)\nSample {idx}")
    fig.colorbar(temp_map, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Multimodal Data Verification\nTime: {times[idx]}", fontsize=14)
    plt.tight_layout()
    plt.show()

    logger.info("="*30)
    

if __name__ == "__main__":
    inspect_multimodal_data()