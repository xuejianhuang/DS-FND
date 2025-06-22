import matplotlib.pyplot as plt
import os

def plot_accuracy_vs_evidence(evidence_nums, acc_dict, output_path):
    """
    Plot accuracy curves of different systems/schedulers across varying numbers of evidence pieces.

    Args:
        evidence_nums (list): List of evidence counts.
        acc_dict (dict): Dictionary mapping method names to their accuracy lists.
        output_path (str): File path to save the plot image.
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
    plt.ylim(90, 96)  # Adjust limits based on your data
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper center', ncol=3, fontsize=10)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    evidence_nums = list(range(11))  # Evidence counts from 0 to 10

    # Twitter
    # dataset="twitter"
    # acc_dict = {
    #     "System2":   [92.7, 93.5, 94.1, 94.4, 95.0, 95.3, 95.1, 94.4, 93.7, 93.3, 93.0],
    #     "DS-FND-PC": [91.4, 91.9, 92.2, 92.4, 92.8, 93.0, 92.9, 92.5, 92.1, 91.9, 91.8],
    #     "DS-FND-TC": [91.6, 92.0, 92.2, 92.5, 92.9, 93.2, 93.1, 92.7, 92.3, 92.1, 91.9],
    # }

    # Weibo
    dataset="weibo"
    acc_dict = {
        "System2":    [92.2, 93.0, 93.5, 94.1, 94.6, 94.8, 94.4, 93.7, 93.2, 92.8, 92.5],
        "DS-FND-PC": [91.3, 91.8, 92.1, 92.3, 92.6, 92.7, 92.5, 92.3, 92.0, 91.7, 91.6],
        "DS-FND-TC": [91.1, 91.7, 92.0, 92.3, 92.5, 92.6, 92.4, 92.1, 91.9, 91.7, 91.5],
    }

    output_path = f"../outputs/{dataset}_evidence_accuracy_curve.png"
    plot_accuracy_vs_evidence(evidence_nums, acc_dict, output_path)
