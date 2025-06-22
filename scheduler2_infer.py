import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import System2Dataset
from model.system1 import System1
from model.system2 import System2
from model.scheduler import Scheduler
import config
import time
from collate_fn import collate_sys2
from utils.util import parse_arguments, split_dataset, index_batch, set_torch_seed
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassF1Score, MulticlassConfusionMatrix
)

def run_scheduler_system_inference(
    scheduler_model,
    system1_model,
    system2_model,
    acc_metric,
    pre_metric,
    rec_metric,
    f1_metric,
    confusion,
    dataloader,
    output_csv: str,
    output_png: str
):
    """
    Perform inference using the scheduler to decide whether to use System1 or System2 for classification,
    and record the classification results.

    Args:
        scheduler_model: The dispatcher model that selects between System1 and System2.
        system1_model: The System1 model.
        system2_model: The System2 model.
        acc_metric, pre_metric, rec_metric, f1_metric, confusion: Metrics for evaluation.
        dataloader: Test data loader.
        output_csv (str): Path to save misclassified samples CSV.
        output_png (str): Path to save confusion matrix image.

    Returns:
        matplotlib.figure.Figure: Confusion matrix figure.
    """
    start_time = time.time()
    scheduler_model.eval()
    system1_model.eval()
    system2_model.eval()

    for metric in [acc_metric, pre_metric, rec_metric, f1_metric, confusion]:
        metric.reset()

    misclassified_samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="scheduler1"):
            labels = batch[0]
            keys = batch[-1]
            scheduler_inputs = batch[1:3]
            system1_inputs = batch[1:6]
            system2_inputs = batch[1:-1]

            dispatch_logits = scheduler_model(scheduler_inputs)
            dispatch_preds = torch.argmax(dispatch_logits, dim=1)

            final_preds = torch.zeros_like(dispatch_preds)

            # System1 branch
            sys1_idx = (dispatch_preds == 0).nonzero(as_tuple=True)[0]
            if sys1_idx.numel() > 0:
                sys1_batch = index_batch(system1_inputs, sys1_idx)
                sys1_output = system1_model(sys1_batch)
                final_preds[sys1_idx] = torch.argmax(sys1_output, dim=1)

            # System2 branch
            sys2_idx = (dispatch_preds == 1).nonzero(as_tuple=True)[0]
            if sys2_idx.numel() > 0:
                sys2_batch = index_batch(system2_inputs, sys2_idx)
                sys2_output = system2_model(sys2_batch)
                final_preds[sys2_idx] = torch.argmax(sys2_output, dim=1)

            # Update metrics
            for metric in [acc_metric, pre_metric, rec_metric, f1_metric, confusion]:
                metric.update(final_preds, labels)

            # Collect misclassified samples
            for key, pred, label, d_pred in zip(keys, final_preds, labels, dispatch_preds):
                if pred != label:
                    misclassified_samples.append([
                        key, label.item(), pred.item(), d_pred.item()
                    ])

    # Save misclassified samples to CSV
    df = pd.DataFrame(misclassified_samples,
                      columns=['Sample Key', 'True Value', 'Predicted Value', 'Scheduler Output'])
    df.to_csv(output_csv, index=False)

    # Print results
    print('----------------- Test Results -----------------')
    print(f'Accuracy:  {acc_metric.compute().cpu().numpy()}')
    print(f'Precision: {pre_metric.compute().cpu().numpy()}')
    print(f'Recall:    {rec_metric.compute().cpu().numpy()}')
    print(f'F1-Score:  {f1_metric.compute().cpu().numpy()}')
    print(f'Confusion Matrix:\n{confusion.compute().cpu().numpy()}')

    fig_, ax_ = confusion.plot()
    fig_.savefig(output_png)

    end_time = time.time()
    print(f"The running time is: {end_time - start_time:.1f} s")


if __name__ == "__main__":
    args = parse_arguments()
    set_torch_seed(args.seed)

    # Load models
    scheduler_model = Scheduler().to(config.device)
    system1_model = System1().to(config.device)
    system2_model = System2(
        config.node_feats, config.edge_feats, config.out_feats,
        config.num_heads, config.n_layers
    ).to(config.device)

    # Load saved model weights
    model_prefix = os.path.join(config.model_saved_path, args.dataset)
    scheduler_model.load_state_dict(torch.load(f"{model_prefix}_Scheduler.pkl"))
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

    run_scheduler_system_inference(
        scheduler_model, system1_model, system2_model,
        acc_metric, pre_metric, rec_metric, f1_metric, confusion,
        test_loader, output_csv, output_png
    )
