import matplotlib.pyplot as plt


def plot_accuracy_vs_evidence(evidence_nums, acc_dict, output_path):
    """
    绘制不同证据数量下，不同系统/调度策略的准确率变化曲线。

    Args:
        evidence_nums (list): 证据数量列表。
        acc_dict (dict): 各方法的准确率数据，键为方法名称，值为准确率列表。
        output_path (str): 图像保存路径。
    """
    plt.figure(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', '*', 'x']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan']

    for i, (label, acc) in enumerate(acc_dict.items()):
        plt.plot(
            evidence_nums,
            acc,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=label,
            linewidth=2
        )

    plt.title('Accuracy vs. Number of Evidence Pieces', fontsize=14)
    plt.xlabel('Number of Evidence', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(evidence_nums)
    plt.ylim(90, 96)  # 自动适应或根据数据调整
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper center', ncol=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    evidence_nums = list(range(11))  # 0-10

    # Twitter 示例数据
    # dataset="twitter"
    # acc_dict = {
    #     "System2":   [92.7, 93.5, 94.1, 94.4, 95.0, 95.3, 95.1, 94.4, 93.7, 93.3, 93.0],
    #     "Scheduler1": [91.4, 91.9, 92.2, 92.4, 92.8, 93.0, 92.9, 92.5, 92.1, 91.9, 91.8],
    #     "Scheduler2": [91.6, 92.0, 92.2, 92.5, 92.9, 93.2, 93.1, 92.7, 92.3, 92.1, 91.9],
    # }

    # Twitter 示例数据
    dataset="webio"
    acc_dict = {
        "System2":    [92.2, 93.0, 93.5, 94.1, 94.6, 94.8, 94.4, 93.7, 93.2, 92.8, 92.5],
        "Scheduler1": [91.3, 91.8, 92.1, 92.3, 92.6, 92.7, 92.5, 92.3, 92.0, 91.7, 91.6],
        "Scheduler2": [91.1, 91.7, 92.0, 92.3, 92.5, 92.6, 92.4, 92.1, 91.9, 91.7, 91.5],
    }





    output_path = f"../outputs/{dataset}_evidence_accuracy_curve.png"
    plot_accuracy_vs_evidence(evidence_nums, acc_dict, output_path)
