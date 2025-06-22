import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(matrix, dataset, method, output_dir="../outputs"):
    """
    Plots and saves a heatmap for a given confusion matrix.

    Args:
        matrix (np.ndarray): Normalized confusion matrix (shape: [num_classes, num_classes]).
        dataset (str): Dataset name (e.g., 'weibo', 'twitter').
        method (str): Model or method name (e.g., 'System1', 'System2').
        output_dir (str): Directory to save the output image.
    """
    plt.figure(figsize=(10, 8))

    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Reds", cbar=True,
                annot_kws={"size": 30},
                xticklabels=['Real News', 'Fake News', 'Unverified News'],
                yticklabels=['Real News', 'Fake News', 'Unverified News'])

    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_{method}_confusion.png")
    plt.savefig(output_path, dpi=600)
    print(f"Confusion matrix saved to {output_path}")
    plt.show()




if __name__ == '__main__':
    # twitter
    # dataset='twitter'
    # method ='System1'
    # confusion_matrix = np.array([[0.935, 0.019, 0.047],
    #                              [0.056, 0.796, 0.148],
    #                              [0.024, 0.042, 0.934]])
    # method ='System2'
    # confusion_matrix = np.array([[0.967, 0.023, 0.009],
    #                              [0.000, 0.926, 0.074],
    #                              [0.003, 0.031, 0.965]])

    # weibo
    dataset = 'weibo'
    # method ='System1'
    # confusion_matrix = np.array([[0.932,0.016,0.052],
    #                              [0.052,0.794,0.155],
    #                              [0.030,0.046,0.924]])
    method  = 'System2'
    confusion_matrix = np.array([[0.968,0.012,0.020],
                                 [0.032,0.948,0.019],
                                 [0.011,0.063,0.926]])

    plot_confusion_matrix(confusion_matrix, dataset, method)
