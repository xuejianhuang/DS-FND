import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Dataset selection: "weibo" or "twitter"
    dataset = 'weibo'

    # Thresholds for system switching
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Accuracy (%) and inference time (ms) corresponding to each threshold
    accuracy = [89.0, 90.4, 90.8, 92.0, 92.2, 92.7, 93.5, 93.7, 94.2, 94.4]
    inference_time = [264, 290, 335, 354, 389, 428, 501, 577, 631, 696]

    # Twitter
    # dataset='twitter'
    # thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # accuracy = [89.6, 90.5, 91.2, 91.7, 92.5, 93.0, 93.3, 93.7, 94.5, 95.0]
    # inference_time = [264, 290, 335, 354, 389, 428, 501, 577, 631, 696]

    # Compute efficiency as normalized inverse of inference time
    min_time = min(inference_time)
    max_time = max(inference_time)
    efficiency = [(max_time - t) / (max_time - min_time) for t in inference_time]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Accuracy on left Y-axis
    color_acc = 'tab:blue'
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12)
    l1 = ax1.plot(thresholds, accuracy, color=color_acc, marker='o', linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(88, 95)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # Plot Efficiency on right Y-axis
    ax2 = ax1.twinx()
    color_eff = 'tab:green'
    ax2.set_ylabel('Efficiency (Normalized)', color=color_eff, fontsize=12)
    l2 = ax2.plot(thresholds, efficiency, color=color_eff, marker='s', linewidth=2, label='Efficiency')
    ax2.tick_params(axis='y', labelcolor=color_eff)
    ax2.set_ylim(0, 1)

    # Combine legends
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', ncol=2, fontsize=10)

    # Save and show plot
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_threshold_vs_accuracy_time.png")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved threshold vs. accuracy/efficiency plot to: {output_path}")
    plt.show()