import os
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config

from model.system1 import System1
from model.system2 import System2
from dataset import System2Dataset
from collate_fn import collate_sys2
from utils.util import split_dataset

def generate_dispatch_labels(sample_keys, sys1_preds, sys2_preds, labels, output_csv_path):
    """
    Generate dispatch labels and save to CSV.
    Each row format: [sample_key, system1_correct, system2_correct, dispatch_label]
    dispatch_label: 0 -> System1, 1 -> System2
    """
    assert len(sample_keys) == len(sys1_preds) == len(sys2_preds) == len(labels)

    rows = [["sample_key", "system1_correct", "system2_correct", "dispatch_label"]]
    for key, s1_pred, s2_pred, true_label in zip(sample_keys, sys1_preds, sys2_preds, labels):
        s1_correct = int(s1_pred == true_label)
        s2_correct = int(s2_pred == true_label)

        if s1_correct and s2_correct:
            dispatch_label = 0  # Both correct, choose System1
        elif not s1_correct and s2_correct:
            dispatch_label = 1  # Only System2 correct
        elif s1_correct and not s2_correct:
            dispatch_label = 0  # Only System1 correct
        else:
            dispatch_label = 0  # Both incorrect, default to System1

        rows.append([key, s1_correct, s2_correct, dispatch_label])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Dispatch labels saved to: {output_csv_path}")


def generate_training_samples(dataloader, sys1_model, sys2_model):
    """
    Run System1 and System2 on all samples to collect predictions and true labels.
    """
    sys1_model.eval()
    sys2_model.eval()

    sample_keys = []
    all_sys1_preds = []
    all_sys2_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            labels = batch[0]
            keys = batch[-1]

            # Note: Confirm these slices correspond to inputs required by each model
            sys1_inputs = batch[1:-5]  # Adjust according to your data structure
            sys2_inputs = batch[1:-1]

            sys1_output = sys1_model(sys1_inputs)
            sys1_preds = torch.argmax(sys1_output, dim=1)

            sys2_output = sys2_model(sys2_inputs)
            sys2_preds = torch.argmax(sys2_output, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_sys1_preds.extend(sys1_preds.cpu().numpy())
            all_sys2_preds.extend(sys2_preds.cpu().numpy())
            sample_keys.extend(keys)

    return sample_keys, all_sys1_preds, all_sys2_preds, all_labels


def get_model_and_data(dataset='weibo', batch_size=config.batch_size):
    """
    Load dataset and models.
    """
    print(f"Loading data and models for dataset: {dataset}")
    data_root = config.weibo_dataset_dir if dataset == 'weibo' else config.twitter_dataset_dir
    dataset_path = os.path.join(data_root, 'dataset_items_merged.json')

    all_items, _, _ = split_dataset(dataset_path, train_ratio=1, val_ratio=0)  # Use all data
    dataset = System2Dataset(all_items, data_root)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sys2)

    sys1_model = System1().to(config.device)
    sys2_model = System2(config.node_feats, config.edge_feats, config.out_feats,
                        config.num_heads, config.n_layers).to(config.device)

    return sys1_model, sys2_model, data_loader


if __name__ == "__main__":
    dataset = 'twitter'
    output_csv_path = os.path.join("outputs", f"{dataset}_dispatch_labels.csv")
    sys1_model, sys2_model, data_loader = get_model_and_data(dataset=dataset)

    # Load pretrained weights
    sys1_model_path = os.path.join(config.model_saved_path, f"{dataset}_System1.pkl")
    sys2_model_path = os.path.join(config.model_saved_path, f"{dataset}_System2.pkl")
    sys1_model.load_state_dict(torch.load(sys1_model_path, map_location=config.device))
    sys2_model.load_state_dict(torch.load(sys2_model_path, map_location=config.device))

    # Generate predictions for all samples
    sample_keys, sys1_preds, sys2_preds, gt_labels = generate_training_samples(data_loader, sys1_model, sys2_model)
    # Save dispatch labels
    generate_dispatch_labels(sample_keys, sys1_preds, sys2_preds, gt_labels, output_csv_path)
