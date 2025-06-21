import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassF1Score, MulticlassConfusionMatrix
)

import config
from dataset import System2Dataset
from model.system1 import System1
from model.system2 import System2
from collate_fn import collate_sys2
from utils.util import parse_arguments, split_dataset, index_batch, set_torch_seed

def run_confidence_scheduler_inference(system1_model, system2_model,
                                       acc_metric, pre_metric, rec_metric, f1_metric, confusion,
                                       dataloader, output_csv, output_png, threshold: float = 0.7):
    """
    基于系统一置信度的推理调度机制。
    """
    start_time = time.time()
    system1_model.eval()
    system2_model.eval()

    for metric in [acc_metric, pre_metric, rec_metric, f1_metric, confusion]:
        metric.reset()

    misclassified_samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="confidence-scheduler"):
            labels = batch[0]
            keys = batch[-1]
            system1_inputs = batch[1:6]
            system2_inputs = batch[1:-1]

            # System 1 推理与置信度计算
            sys1_output = system1_model(system1_inputs)
            sys1_probs = torch.softmax(sys1_output, dim=1)
            sys1_conf, sys1_preds = torch.max(sys1_probs, dim=1)

            final_preds = sys1_preds.clone()
            low_conf_mask = sys1_conf < threshold

            # 对低置信度样本调用系统2
            if low_conf_mask.any():
                sys2_idx = low_conf_mask.nonzero(as_tuple=True)[0]
                sys2_batch = index_batch(system2_inputs, sys2_idx)
                sys2_output = system2_model(sys2_batch)
                sys2_preds = torch.argmax(sys2_output, dim=1)
                final_preds[sys2_idx] = sys2_preds

            # 更新评估指标
            for metric in [acc_metric, pre_metric, rec_metric, f1_metric, confusion]:
                metric.update(final_preds, labels)

            # 记录分类错误样本
            for key, pred, label, conf in zip(keys, final_preds, labels, sys1_conf):
                if pred != label:
                    misclassified_samples.append([
                        key, label.item(), pred.item(), round(conf.item(), 4)
                    ])

    # 保存错误样本
    pd.DataFrame(misclassified_samples, columns=['Sample Key', 'True Label', 'Predicted Label', 'System1 Confidence'])\
        .to_csv(output_csv, index=False)

    # 输出评估结果
    print('\n----------- Confidence Scheduler Results -----------')
    print(f'Accuracy:  {acc_metric.compute().cpu().numpy()}')
    print(f'Precision: {pre_metric.compute().cpu().numpy()}')
    print(f'Recall:    {rec_metric.compute().cpu().numpy()}')
    print(f'F1-Score:  {f1_metric.compute().cpu().numpy()}')
    print(f'Confusion Matrix:\n{confusion.compute().cpu().numpy()}')

    fig, ax = confusion.plot()
    fig.savefig(output_png)

    print(f"Total runtime: {time.time() - start_time:.1f} s")


if __name__ == "__main__":
    args = parse_arguments()
    set_torch_seed(args.seed)

    # 模型加载
    system1_model = System1().to(config.device)
    system2_model = System2(
        config.node_feats, config.edge_feats, config.out_feats,
        config.num_heads, config.n_layers
    ).to(config.device)

    model_prefix = os.path.join(config.model_saved_path, args.dataset)
    system1_model.load_state_dict(torch.load(f"{model_prefix}_System1.pkl"))
    system2_model.load_state_dict(torch.load(f"{model_prefix}_System2.pkl"))

    # 加载数据集
    data_root = config.weibo_dataset_dir if args.dataset == 'weibo' else config.twitter_dataset_dir
    dataset_path = os.path.join(data_root, 'dataset_items_merged.json')
    _, _, test_items = split_dataset(dataset_path, config.train_ratio, config.val_ratio)
    test_dataset = System2Dataset(test_items, data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_sys2)

    # 创建输出目录
    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)
    output_csv = os.path.join(save_dir, f"{args.dataset}_cooperation_misclassified.csv")
    output_png = os.path.join(save_dir, f"{args.dataset}_cooperation_confusion_matrix.png")

    # 初始化评估指标
    acc_metric = MulticlassAccuracy(num_classes=config.num_classes, average='macro').to(config.device)
    pre_metric = MulticlassPrecision(num_classes=config.num_classes, average=None).to(config.device)
    rec_metric = MulticlassRecall(num_classes=config.num_classes, average=None).to(config.device)
    f1_metric = MulticlassF1Score(num_classes=config.num_classes, average=None).to(config.device)
    confusion = MulticlassConfusionMatrix(num_classes=config.num_classes).to(config.device)

    run_confidence_scheduler_inference(system1_model, system2_model,
                                       acc_metric, pre_metric, rec_metric, f1_metric, confusion,
                                       test_loader, output_csv, output_png,threshold=0.5)
