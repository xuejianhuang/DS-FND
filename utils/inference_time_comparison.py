import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Method names and their corresponding average inference times (in milliseconds)
methods = [
    'GPT-4', 'GTP-4V', 'GTP-4V+search',
    'LSTM_Word2vec', 'TDRD', 'DC-CNN', 'DEDA',
    'MCAN', 'HMCAN', 'MTTV', 'ITS', 'HGA-MMRD', 'MMCAN', 'HESN',
    'CCN', 'DEETSA',
    'System1', 'System2', 'DS-FND-PC', 'DS-FND-TC'
]

times = [
    1253, 1721, 2545,
    121, 134, 130, 152,
    321, 334, 424, 343, 467, 480, 475,
    524, 546,
    220, 696, 428, 410
]

if __name__ == '__main__':
    # Sort by time ascending for top-down view
    sorted_data = sorted(zip(times, methods))
    times, methods = zip(*sorted_data)

    # Set color palette
    colors = cm.tab20(np.linspace(0, 1, len(methods)))

    # Create plot
    plt.figure(figsize=(10, 10))
    bars = plt.barh(methods, times, color=colors, edgecolor='black')

    # Add time labels next to each bar
    for bar, time in zip(bars, times):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f'{time} ms', va='center', fontsize=10)

    # Style settings
    plt.xlabel("Average Inference Time (ms)", fontsize=12)
    plt.ylabel("Methods", fontsize=12)
    plt.title("Average Inference Time of Different Methods", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save output
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "inference_time.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved inference time chart to: {output_path}")
    plt.show()
