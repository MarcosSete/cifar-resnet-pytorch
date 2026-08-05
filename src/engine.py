"""
src/engine.py

As funções que rodam uma época de treino e uma época de avaliação.
Separadas de train.py pra poderem ser testadas isoladamente (é isso que
permite o "overfit a single batch" test em tests/test_model.py).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import AverageMeter


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
) -> dict:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(acc, n=images.size(0))

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(acc, n=images.size(0))

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "predictions": torch.cat(all_preds),
        "labels": torch.cat(all_labels),
    }
