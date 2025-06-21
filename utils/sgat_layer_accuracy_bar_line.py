import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # SGAT 层数
    layers = [1, 2, 3]

    # 准确率数据
    #twitter
    # dataset="twitter"
    # system2_acc = [94.6, 95.3, 95.0]
    # scheduler1_acc = [92.6, 93.0, 92.8]
    # scheduler2_acc = [92.7, 93.2, 93.0]

    #weibo
    dataset="weibo"
    system2_acc = [94.1, 94.8, 94.2]
    scheduler1_acc = [92.3, 92.7, 92.4]
    scheduler2_acc = [92.2, 93.2, 92.3]

    # 设置柱状图参数
    bar_width = 0.25
    x = np.arange(len(layers))  # x轴位置

    # 创建图形
    plt.figure(figsize=(9, 6))

    # 柱状图
    plt.bar(x - bar_width, system2_acc, width=bar_width, label='System2', color='skyblue')
    plt.bar(x, scheduler1_acc, width=bar_width, label='Scheduler1', color='orange')
    plt.bar(x + bar_width, scheduler2_acc, width=bar_width, label='Scheduler2', color='mediumseagreen')

    # 折线图叠加
    # plt.plot(x, system2_acc, color='blue', linestyle='--', marker='o')
    # plt.plot(x, scheduler1_acc, color='red', linestyle='--', marker='s')
    # plt.plot(x, scheduler2_acc, color='green', linestyle='--', marker='^')

    # 设置坐标轴与标签
    plt.xlabel('Number of SGAT Layers', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy of Different Methods with Varying SGAT Layers', fontsize=14)
    plt.xticks(x, layers)
    plt.ylim(92, 96)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 图例
    plt.legend(loc='upper center', ncol=3)

    # 显示数值注释
    # for i in range(len(layers)):
    #     plt.text(x[i] - bar_width, system2_acc[i] + 0.1, f'{system2_acc[i]}', ha='center', fontsize=9)
    #     plt.text(x[i], scheduler1_acc[i] + 0.1, f'{scheduler1_acc[i]}', ha='center', fontsize=9)
    #     plt.text(x[i] + bar_width, scheduler2_acc[i] + 0.1, f'{scheduler2_acc[i]}', ha='center', fontsize=9)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"../outputs/{dataset}_sgat_layer_accuracy_bar_line.png", dpi=300)
    plt.show()
