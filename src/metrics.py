"""
src/metrics.py

Métricas além de acurácia simples — importante porque acurácia agregada
esconde desbalanceamento de erro entre classes (ex: o modelo pode acertar
90% no geral mas confundir sistematicamente "cat" com "dog").
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix


def compute_confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor):
    return confusion_matrix(labels.numpy(), predictions.numpy())


def compute_classification_report(predictions: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> str:
    return classification_report(
        labels.numpy(), predictions.numpy(), target_names=class_names, digits=3
    )
