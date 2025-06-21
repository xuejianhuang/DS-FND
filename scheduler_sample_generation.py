import os
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import config

from model.system1 import System1
from model.system2 import System2
from dataset import System2Dataset
from collate_fn import collate_sys2
from utils.util import (
    split_dataset
)

def generate_dispatch_labels(sample_keys, system1_preds, system2_preds, labels, output_csv_path):
    """
    构建调度标签并保存为CSV
    每行格式：[sample_key, system1_correct, system2_correct, dispatch_label]
    dispatch_label: 0 -> System1, 1 -> System2
    """
    assert len(sample_keys) == len(system1_preds) == len(system2_preds) == len(labels)

    rows = [["sample_key", "system1_correct", "system2_correct", "dispatch_label"]]
    for key, s1, s2, gt in zip(sample_keys, system1_preds, system2_preds, labels):
        s1_correct = int(s1 == gt)
        s2_correct = int(s2 == gt)

        if s1_correct and s2_correct:
            dispatch_label = 0  # 都对，选System1
        elif not s1_correct and s2_correct:
            dispatch_label = 1  # System2正确
        elif s1_correct and not s2_correct:
            dispatch_label = 0  # System1正确
        else:
            dispatch_label = 0  # 都错，默认System1

        rows.append([key, s1_correct, s2_correct, dispatch_label])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"调度标签样本已保存到：{output_csv_path}")


def generate_training_samples(dataloader, sys1_model, sys2_model):
    """
    对所有样本运行System1和System2，收集预测与真实标签
    """
    sys1_model.eval()
    sys2_model.eval()

    sample_keys, all_s1_preds, all_s2_preds, all_labels = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            labels = batch[0]
            keys = batch[-1]

            sys1_inputs = batch[1:-5]
            sys2_inputs = batch[1:-1]

            # System1 预测
            s1_output = sys1_model(sys1_inputs)
            s1_preds = torch.argmax(s1_output, dim=1)

            # System2 预测
            s2_output = sys2_model(sys2_inputs)
            s2_preds = torch.argmax(s2_output, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_s1_preds.extend(s1_preds.cpu().numpy())
            all_s2_preds.extend(s2_preds.cpu().numpy())
            sample_keys.extend(keys)

    return sample_keys, all_s1_preds, all_s2_preds, all_labels

def get_model_and_data(dataset='weibo', batch=config.batch_size):
    """
    加载数据集与模型
    """
    print(f"加载数据与模型：{dataset}")
    data_root_dir = config.weibo_dataset_dir if dataset == 'weibo' else config.twitter_dataset_dir
    dataset_path = os.path.join(data_root_dir, 'dataset_items_merged.json')

    all_items, _, _ = split_dataset(dataset_path, 1, 0)  # 使用全部数据
    ds = System2Dataset(all_items, data_root_dir)
    data_loader = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_sys2)

    sys1_model = System1().to(config.device)
    sys2_model = System2(config.node_feats, config.edge_feats, config.out_feats,
                         config.num_heads, config.n_layers).to(config.device)

    return sys1_model, sys2_model, data_loader

if __name__ == "__main__":
    dataset = 'twitter'
    output_csv_path = os.path.join("outputs", f"{dataset}_dispatch_labels.csv")
    sys1_model, sys2_model, data_loader = get_model_and_data(dataset=dataset)
    # 加载预训练权重
    sys1_model_path = os.path.join(config.model_saved_path, f"{dataset}_System1.pkl")
    sys2_model_path = os.path.join(config.model_saved_path, f"{dataset}_System2.pkl")
    sys1_model.load_state_dict(torch.load(sys1_model_path, map_location=config.device))
    sys2_model.load_state_dict(torch.load(sys2_model_path, map_location=config.device))

    # 生成预测结果
    sample_keys, sys1_preds, sys2_preds, gt_labels = generate_training_samples(data_loader, sys1_model, sys2_model)
    # 保存调度标签
    generate_dispatch_labels(sample_keys, sys1_preds, sys2_preds, gt_labels, output_csv_path)


