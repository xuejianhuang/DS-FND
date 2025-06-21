import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 方法名与对应平均推理时间
methods = [
    'GPT-4', 'GTP-4V', 'GTP-4V+search',
    'LSTM_Word2vec', 'TDRD', 'DC-CNN', 'DEDA',
    'MCAN', 'HMCAN', 'MTTV', 'ITS', 'HGA-MMRD', 'MMCAN', 'HESN',
    'CCN', 'DEETSA',
    'System1', 'System2', 'Scheduler1', 'Scheduler2'
]

times = [
    1253, 1721, 2545,
    121, 134, 130, 152,
    321, 334, 424, 343, 467, 480, 475,
    524, 546,
    220, 696, 428, 410
]
if __name__ == '__main__':
    # 倒序排列，使得最小时间在顶部
    methods = methods[::-1]
    times = times[::-1]

    # 设置不同颜色
    colors = cm.tab20(np.linspace(0, 1, len(methods)))

    # 创建图形
    plt.figure(figsize=(10, 9))
    plt.barh(methods, times, color=colors, edgecolor='black')

    # 设置图表样式
    plt.xlabel("Average Inference Time (ms)")
    plt.ylabel("Methods")
    plt.title("Average Inference Time of Different Methods")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存并显示图像
    plt.savefig("../outputs/inference_time.png", dpi=300)
    plt.show()
