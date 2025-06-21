import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体字体
#plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体字体
if __name__ == '__main__':
    # twitter
    # dataset='twitter'
    # methond='System1'
    # confusion_matrix = np.array([[0.935, 0.019, 0.047],
    #                              [0.056, 0.796, 0.148],
    #                              [0.024, 0.042, 0.934]])
    # methond='System2'
    # confusion_matrix = np.array([[0.967, 0.023, 0.009],
    #                              [0.000, 0.926, 0.074],
    #                              [0.003, 0.031, 0.965]])

    # weibo
    dataset='weibo'
    # methond='System1'
    # confusion_matrix = np.array([[0.932,0.016,0.052],
    #                              [0.052,0.794,0.155],
    #                              [0.030,0.046,0.924]])
    methond = 'System2'
    confusion_matrix = np.array([[0.968,0.012,0.020],
                                 [0.032,0.948,0.019],
                                 [0.011,0.063,0.926]])


    # 创建一个热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True,fmt=".3f", cmap="Reds", cbar=True,
                annot_kws={"size": 30},  # 设置热力图上数字的大小
                xticklabels=['Real News', 'Fake News', 'Unverified News'],
                yticklabels=['Real News', 'Fake News', 'Unverified News'])

    # 添加标题和标签
    #plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label',fontsize=18)
    plt.ylabel('True Label',fontsize=18)

    # 设置xticklabels和yticklabels的大小
    plt.xticks(fontsize=16)  # 设置x轴标签大小
    plt.yticks(fontsize=16)  # 设置y轴标签大小
   #plt.savefig('weibo_confusion.png', dpi=600)
    plt.savefig(f'../outputs/{dataset}_{methond}_confusion.png', dpi=600)

    plt.show()
