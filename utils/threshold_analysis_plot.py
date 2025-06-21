import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Weibo
    dataset = 'weibo'
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    accuracy = [89.0, 90.4, 90.8, 92.0, 92.2, 92.7, 93.5, 93.7, 94.2, 94.4]
    inference_time = [264, 290, 335, 354, 389, 428, 501, 577, 631, 696]

    # Twitter
    # dataset='twitter'
    # thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # accuracy = [89.6, 90.5, 91.2, 91.7, 92.5, 93.0, 93.3, 93.7, 94.5, 95.0]
    # inference_time = [264, 290, 335, 354, 389, 428, 501, 577, 631, 696]

    # 推理效率：Min-Max归一化 (越小的时间效率越高)
    min_it = min(inference_time)
    max_it = max(inference_time)
    efficiency = [(max_it - t) / (max_it - min_it) for t in inference_time]

    # 绘图
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左 Y 轴：准确率
    color_acc = 'tab:blue'
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Accuracy (%)', color=color_acc)
    l1 = ax1.plot(thresholds, accuracy, color=color_acc, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(88, 95)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # 右 Y 轴：效率（归一化）
    ax2 = ax1.twinx()
    color_eff = 'tab:green'
    ax2.set_ylabel('Efficiency (Normalized)', color=color_eff)
    l2 = ax2.plot(thresholds, efficiency, color=color_eff, marker='s', label='Efficiency')
    ax2.tick_params(axis='y', labelcolor=color_eff)
    ax2.set_ylim(0, 1)

    # 合并图例
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', ncol=2)

    # 标题和布局
    #plt.title('Accuracy and Efficiency vs Threshold')
    plt.tight_layout()
    fig.savefig(f"../outputs/{dataset}_threshold_vs_accuracy_time.png", dpi=300)
    plt.show()
