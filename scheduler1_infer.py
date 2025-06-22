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
    Inference with confidence-based scheduler:
    Use System1 predictions if confidence exceeds threshold,
    otherwise fallback to System2 predictions.

    Args:
        system1_model (nn.Module): The first-stage model.
        system2_model (nn.Module): The second-stage model.
        acc_metric, pre_metric, rec_metric, f1_metric, confusion: torchmetrics for evaluation.
        dataloader (DataLoader): Test dataset loader.
        output_csv (str): Path to save misclassified samples CSV.
        output_png (str): Path to save confusion matrix image.
        threshold (float): Confidence threshold to switch from System1 to System2.

    Returns:
        None
    """
    start_time = time.time()
    system1_model.eval()
    system2_model.eval()

    for metric in [acc_metric, pre_metric, rec_metric, f1_metric, confusion]:
        metric.reset()

    misclassified_samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Confidence Scheduler Inference"):
            labels = batch[0]
            keys = batch[-1]
            system1_inputs = batch[1:6]
            system2_inputs = batch[1:-1]

            # System1 inference and confidence calculation
            sys1_output = system1_model(system1_inputs)
            sys1_probs = torch.softmax(sys1_output, dim=1)
            sys1_conf, sys1_preds = torch.max(sys1_probs, dim=1)

            final_preds = sys1_preds.clone()
            low_conf_mask = sys1_conf < threshold

            # For low confidence samples, use System2 predictions
            if low_conf_mask.any():
                sys2_indices = low_conf_mask.nonzero(as_tuple=True)[0]
                sys2_batch = index_batch(system2_inputs, sys2_indices)
                sys2_output = system2_model(sys2_batch)
                sys2_preds = torch.argmax(sys2_output, dim=1)
                final_preds[sys2_indices] = sys2_preds

            # Update metrics
            for metric in [acc_metric, pre_metric, rec_metric, f1_metric, confusion]:
                metric.update(final_preds, labels)

            # Collect misclassified samples with System1 confidence
            for key, pred, label, conf in zip(keys, final_preds, labels, sys1_conf):
                if pred != label:
                    misclassified_samples.append([key, label.item(), pred.item(), round(conf.item(), 4)])

    # Save misclassified samples to CSV
    pd.DataFrame(
        misclassified_samples,
        columns=['Sample Key', 'True Label', 'Predicted Label', 'System1 Confidence']
    ).to_csv(output_csv, index=False)

    # Print evaluation results
    print('\n----------- Confidence Scheduler Results -----------')
    print(f'Accuracy:  {acc_metric.compute().cpu().numpy()}')
    print(f'Precision: {pre_metric.compute().cpu().numpy()}')
    print(f'Recall:    {rec_metric.compute().cpu().numpy()}')
    print(f'F1-Score:  {f1_metric.compute().cpu().numpy()}')
    print(f'Confusion Matrix:\n{confusion.compute().cpu().numpy()}')

    fig, ax = confusion.plot()
    fig.savefig(output_png)

    print(f"Total runtime: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    args = parse_arguments()
    set_torch_seed(args.seed)

    # Load models
    system1_model = System1().to(config.device)
    system2_model = System2(
        config.node_feats, config.edge_feats, config.out_feats,
        config.num_heads, config.n_layers
    ).to(config.device)

    model_prefix = os.path.join(config.model_saved_path, args.dataset)
    system1_model.load_state_dict(torch.load(f"{model_prefix}_System1.pkl"))
    system2_model.load_state_dict(torch.load(f"{model_prefix}_System2.pkl"))

    # Load dataset
    data_root = config.weibo_dataset_dir if args.dataset == 'weibo' else config.twitter_dataset_dir
    dataset_path = os.path.join(data_root, 'dataset_items_merged.json')
    _, _, test_items = split_dataset(dataset_path, config.train_ratio, config.val_ratio)
    test_dataset = System2Dataset(test_items, data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_sys2)

    # Create output directory
    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)
    output_csv = os.path.join(save_dir, f"{args.dataset}_cooperation_misclassified.csv")
    output_png = os.path.join(save_dir, f"{args.dataset}_cooperation_confusion_matrix.png")

    # Initialize metrics
    acc_metric = MulticlassAccuracy(num_classes=config.num_classes, average='macro').to(config.device)
    pre_metric = MulticlassPrecision(num_classes=config.num_classes, average=None).to(config.device)
    rec_metric = MulticlassRecall(num_classes=config.num_classes, average=None).to(config.device)
    f1_metric = MulticlassF1Score(num_classes=config.num_classes, average=None).to(config.device)
    confusion = MulticlassConfusionMatrix(num_classes=config.num_classes).to(config.device)

    run_confidence_scheduler_inference(
        system1_model, system2_model,
        acc_metric, pre_metric, rec_metric, f1_metric, confusion,
        test_loader, output_csv, output_png, threshold=0.5
    )
