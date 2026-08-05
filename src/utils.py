"""
src/utils.py

Utilitários compartilhados entre treino e avaliação.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixa todas as fontes de aleatoriedade relevantes. Sem isso, dois
    treinos com o mesmo config podem dar resultados diferentes, o que
    torna impossível comparar experimentos de forma confiável."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Acumula uma métrica (loss, acurácia) ao longo dos batches de uma
    época e devolve a média — padrão em qualquer training loop sério."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def save_checkpoint(model: torch.nn.Module, optimizer, epoch: int, best_val_acc: float, path: str) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
