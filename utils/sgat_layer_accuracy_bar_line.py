import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Dataset name
    dataset = "weibo"

    # SGAT layer configurations
    layers = [1, 2, 3]

    # Accuracy for different methods (in %)
    system2_acc = [94.1, 94.8, 94.2]
    scheduler1_acc = [92.3, 92.7, 92.4]
    scheduler2_acc = [92.2, 93.2, 92.3]

    # dataset="twitter"
    # system2_acc = [94.6, 95.3, 95.0]
    # scheduler1_acc = [92.6, 93.0, 92.8]
    # scheduler2_acc = [92.7, 93.2, 93.0]

    # Bar chart parameters
    bar_width = 0.25
    x = np.arange(len(layers))  # x-axis positions

    # Create figure
    plt.figure(figsize=(9, 6))

    # Plot bar charts
    bars1 = plt.bar(x - bar_width, system2_acc, width=bar_width, label='System2', color='skyblue')
    bars2 = plt.bar(x, scheduler1_acc, width=bar_width, label='DS-FND-PC', color='orange')
    bars3 = plt.bar(x + bar_width, scheduler2_acc, width=bar_width, label='DS-FND-TC', color='mediumseagreen')

    # Annotate each bar with its value
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.1f}',
                     ha='center', va='bottom', fontsize=9)

    # Axis labels and title
    plt.xlabel('Number of SGAT Layers', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy of Different Methods with Varying SGAT Layers', fontsize=14)
    plt.xticks(x, layers, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(92, 96)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Legend
    plt.legend(loc='upper center', ncol=3, fontsize=10)

    # Save figure
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_sgat_layer_accuracy_bar.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved SGAT layer accuracy bar chart to: {output_path}")
    plt.show()