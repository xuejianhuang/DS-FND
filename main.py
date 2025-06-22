import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)
import config
import time
from utils.util import parse_arguments, set_torch_seed
from logger import set_logger, log_args, log_config
from dataset import get_model_and_data

# =================== Training Function ===================
def train(model, train_dataloader, criterion, optimizer, metric, epoch, writer):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        metric (Metric): Metric to evaluate performance.
        epoch (int): Current epoch number.
        writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
        tuple: (train_accuracy, train_loss)
    """
    model.train()
    loss_list = []
    total_steps = len(train_dataloader)

    for step, batch in enumerate(train_dataloader):
        labels = batch[0]
        data = batch[1:-1]

        optimizer.zero_grad()
        probs = model(data)
        preds = torch.argmax(probs, dim=-1)

        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        metric.update(preds, labels)
        loss_list.append(loss.item())

        if step % 10 == 0:
            acc = metric.compute()
            logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
            global_step = epoch * total_steps + step
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Acc/train', acc, global_step)

    writer.flush()
    train_loss = np.mean(loss_list)
    train_acc = metric.compute().cpu().numpy()
    metric.reset()
    return train_acc, train_loss

# =================== Validation Function ===================
@torch.no_grad()
def val(model, metric, criterion, val_dataloader, epoch, writer):
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        metric (Metric): Metric to evaluate performance.
        criterion (nn.Module): Loss function.
        val_dataloader (DataLoader): DataLoader for validation data.
        epoch (int): Current epoch number.
        writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
        tuple: (validation_accuracy, validation_loss)
    """
    model.eval()
    loss_list = []

    for batch in val_dataloader:
        labels = batch[0]
        data = batch[1:-1]

        probs = model(data)
        preds = torch.argmax(probs, dim=-1)

        loss = criterion(probs, labels)
        loss_list.append(loss.item())
        metric.update(preds, labels)

    val_loss = np.mean(loss_list)
    val_acc = metric.compute().cpu().numpy()
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Acc/val', val_acc, epoch)
    metric.reset()
    return val_acc, val_loss

# =================== Testing / Inference Function ===================
@torch.no_grad()
def inference(model, acc_metric, pre_metric, rec_metric, f1_metric, confusion, test_dataloader, output_csv, output_png):
    """
    Test the model and save misclassified samples to CSV.

    Args:
        model (nn.Module): The model to test.
        acc_metric (Metric): Accuracy metric.
        pre_metric (Metric): Precision metric.
        rec_metric (Metric): Recall metric.
        f1_metric (Metric): F1 score metric.
        confusion (Metric): Confusion matrix metric.
        test_dataloader (DataLoader): DataLoader for test data.
        output_csv (str): File path to save misclassified samples.
        output_png (str): File path to save confusion matrix figure.

    Returns:
        None
    """
    model.eval()

    acc_metric.reset()
    pre_metric.reset()
    rec_metric.reset()
    f1_metric.reset()
    confusion.reset()

    misclassified_samples = []

    for batch in test_dataloader:
        labels = batch[0]
        data = batch[1:-1]
        keys = batch[-1]

        probs = model(data)
        preds = torch.argmax(probs, dim=-1)

        acc_metric.update(preds, labels)
        pre_metric.update(preds, labels)
        rec_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        confusion.update(preds, labels)

        # Collect misclassified samples
        for key, pred, label in zip(keys, preds, labels):
            if pred != label:
                misclassified_samples.append([key, label.item(), pred.item()])

    df = pd.DataFrame(misclassified_samples, columns=['Sample Key', 'True Value', 'Predicted Value'])
    df.to_csv(output_csv, index=False)

    test_acc = acc_metric.compute().cpu().numpy()
    test_pre = pre_metric.compute().cpu().numpy()
    test_rec = rec_metric.compute().cpu().numpy()
    test_f1 = f1_metric.compute().cpu().numpy()
    fig, ax = confusion.plot()
    fig.savefig(output_png)

    confusion_matrix = confusion.compute().cpu().numpy()
    logger.info('--------------------- Test Results -------------------------------')
    logger.info(f'Accuracy: {test_acc:.4f}, Precision: {test_pre}, Recall: {test_rec}, F1 Score: {test_f1}')
    logger.info(f'Confusion Matrix:\n{confusion_matrix}')


if __name__ == '__main__':
    args = parse_arguments()
    set_torch_seed(args.seed)
    logger = set_logger(args)
    log_args(logger, args)
    log_config(logger)

    # Setup classes and weights based on model type
    if args.model == "Scheduler":
        num_classes = 2
        average = 'micro'
        weights = torch.tensor([0.1, 1], dtype=torch.float32).to(config.device)
    else:
        num_classes = 3
        weights = None
        average = 'macro'

    writer = SummaryWriter()

    model, train_dataloader, val_dataloader, test_dataloader = get_model_and_data(
        model_name=args.model, dataset=args.dataset, batch=args.batch
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total number of trainable parameters: {total_params}')

    start_time = time.time()

    criterion = CrossEntropyLoss(weight=weights)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average=average).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=config.decayRate)

    best_acc = 0
    early_stop_cnt = 0

    model_saved_path = os.path.join(config.model_saved_path, f"{args.dataset}_{args.model}.pkl")
    os.makedirs(config.model_saved_path, exist_ok=True)

    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)

    misclassified_save_path = os.path.join(save_dir, f"{args.dataset}_{args.model}_misclassified.csv")
    confusion_matrix_save_path = os.path.join(save_dir, f"{args.dataset}_{args.model}_confusion_matrix.png")

    if args.mode in ['train', 'both']:
        for epoch in range(config.epoch):
            logger.info(f'----------- Epoch: {epoch} -----------')
            train_acc, train_loss = train(model, train_dataloader, criterion, optimizer, acc_metric, epoch, writer)
            logger.info(f'Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.3f}')
            val_acc, val_loss = val(model, acc_metric, criterion, val_dataloader, epoch, writer)
            logger.info(f'Validation Loss: {val_loss:.5f}, Validation Acc: {val_acc:.3f}\n')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_saved_path)
                logger.info(f"Model saved with accuracy: {best_acc:.3f}")
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt >= config.patience:
                logger.info(f"Early stopping triggered after {early_stop_cnt} epochs without improvement.")
                break

            lr_scheduler.step()

    if args.mode in ['test', 'both']:
        pre_metric = MulticlassPrecision(num_classes=num_classes, average=None).to(config.device)
        rec_metric = MulticlassRecall(num_classes=num_classes, average=None).to(config.device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average=None).to(config.device)
        confusion = MulticlassConfusionMatrix(num_classes=num_classes).to(config.device)

        model.load_state_dict(torch.load(model_saved_path))
        inference(
            model, acc_metric, pre_metric, rec_metric, f1_metric, confusion,
            test_dataloader, misclassified_save_path, confusion_matrix_save_path
        )

    writer.close()
    end_time = time.time()
    logger.info(f"Total running time: {end_time - start_time:.1f} seconds")