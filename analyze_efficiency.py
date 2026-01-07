import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
from thop import profile
from copy import deepcopy

# 导入你的三个模型
# 请确保这些文件都在正确的位置，且类名正确
from model.MMMambaModel import MMMambaModel
from model.MMLSTMModel import MMLSTMModel
from model.MMTransformerModel import MMTransformerModel

# ================= 配置 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 168
PRED_LEN = 24
# ⚠️ 测评时 Batch Size 通常设为 1 (测单样本延迟) 或 64 (测吞吐量)
# 这里我们用 1 来模拟实时预测场景，这对论文更有说服力
BATCH_SIZE = 1 

# ================= 测评函数 =================
def measure_model(model, model_name):
    model.eval()
    model.to(DEVICE)
    
    # 1. 创建虚拟输入 (Dummy Input)
    # 形状必须与真实训练时完全一致
    dummy_load = torch.randn(BATCH_SIZE, SEQ_LEN, 1).to(DEVICE)
    dummy_img = torch.randn(BATCH_SIZE, SEQ_LEN, 2, 32, 32).to(DEVICE)
    dummy_text = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN, 32)).to(DEVICE) # 整数索引
    
    print(f"--- 正在分析: {model_name} ---")

    # ---------------------------------------------------
    # 指标 1: 参数量 (Parameters) - 硬件无关
    # ---------------------------------------------------
    params = sum(p.numel() for p in model.parameters()) / 1e6 # 换算成 Million (M)
    
    # ---------------------------------------------------
    # 指标 2: 计算量 (FLOPs) - 硬件无关
    # 使用 thop 库自动计算
    # ---------------------------------------------------
    try:
        # thop 需要输入是在 CPU 还是 GPU 都可以，这里用 dummy inputs
        # 注意：thop 可能无法完美统计自定义 CUDA kernel (如 Mamba)，但能统计 Linear/Conv 等大头
        macs, _ = profile(model, inputs=(dummy_load, dummy_img, dummy_text), verbose=False)
        flops = macs * 2 # 通常 FLOPs ≈ 2 * MACs
        flops = flops / 1e9 # 换算成 Giga (G)
    except Exception as e:
        print(f"Warning: FLOPs calculation failed for {model_name}. Reason: {e}")
        flops = 0

    # ---------------------------------------------------
    # 指标 3: 推理延迟 (Latency) & 吞吐量 (Throughput)
    # 必须预热 (Warm-up) 并在 GPU 上同步时间
    # ---------------------------------------------------
    # 预热 50 次，让 GPU 进入状态
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_load, dummy_img, dummy_text)
    
    # 正式测量 100 次
    iterations = 100
    timings = []
    
    # 使用 CUDA Event 计时才最精准
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in range(iterations):
            starter.record()
            _ = model(dummy_load, dummy_img, dummy_text)
            ender.record()
            # 等待 GPU 完成
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 返回毫秒
            timings.append(curr_time)
            
    mean_latency = np.mean(timings) # 毫秒 (ms)
    std_latency = np.std(timings)
    # 吞吐量 = 1000ms / 平均耗时 * BatchSize
    throughput = (1000 / mean_latency) * BATCH_SIZE 

    # ---------------------------------------------------
    # 指标 4: 峰值显存 (Peak Memory)
    # ---------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_load, dummy_img, dummy_text)
    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB

    return {
        "Model": model_name,
        "Params (M)": f"{params:.2f}",
        "FLOPs (G)": f"{flops:.3f}",
        "Latency (ms)": f"{mean_latency:.2f} ± {std_latency:.2f}",
        "Throughput (sample/s)": f"{throughput:.1f}",
        "Memory (MB)": f"{max_memory:.2f}"
    }

# ================= 主程序 =================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("开始科学效率评估 (Batch Size = 1)...\n")
    
    results = []

    # 1. 实例化三个模型
    # 确保参数与你训练时一致
    mamba_model = MMMambaModel(seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=128, n_layers=2)
    lstm_model = MMLSTMModel(seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=128, n_layers=2)
    trans_model = MMTransformerModel(seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=128, n_layers=2, nhead=4)

    # 2. 依次测评
    results.append(measure_model(mamba_model, "MM-Mamba (Ours)"))
    results.append(measure_model(lstm_model, "MM-LSTM"))
    results.append(measure_model(trans_model, "MM-Transformer"))

    # 3. 生成论文表格
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("FINAL EFFICIENCY REPORT (Ready for LaTeX/Paper)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 导出为 CSV 方便你画图
    df.to_csv("efficiency_report.csv", index=False)
    print("\n结果已保存至 efficiency_report.csv")