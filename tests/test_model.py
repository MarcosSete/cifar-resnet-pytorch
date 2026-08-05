"""
tests/test_model.py

Dois tipos de teste, cobrindo o que realmente importa verificar antes de
gastar tempo/dinheiro rodando um treino completo:

1. Testes de shape: garantem que a arquitetura não tem bug estrutural
   (dimensão errada, conexão residual quebrada).
2. "Overfit a single batch": um sanity check clássico (recomendado por
   Karpathy no "recipe for training neural networks") — se o modelo não
   consegue nem decorar um único batch pequeno, tem bug na loss, no
   otimizador, ou no fluxo de gradiente, e rodar o treino completo seria
   perda de tempo. Isso roda em segundos, sem precisar do CIFAR-10 de
   verdade (uso dados sintéticos aleatórios).

Rode com: PYTHONPATH=. pytest tests/ -v
"""

import torch
import torch.nn as nn

from src.model import ResNetCifar, build_model


def test_output_shape_matches_num_classes():
    model = build_model(num_classes=10)
    x = torch.randn(4, 3, 32, 32)  # batch de 4 imagens CIFAR-10 (3x32x32)
    out = model(x)
    assert out.shape == (4, 10)


def test_works_with_different_batch_sizes():
    model = build_model(num_classes=10)
    for batch_size in [1, 2, 16]:
        x = torch.randn(batch_size, 3, 32, 32)
        out = model(x)
        assert out.shape == (batch_size, 10)


def test_custom_num_classes():
    model = ResNetCifar(num_classes=100)  # ex: CIFAR-100
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_gradients_flow_through_all_parameters():
    """Garante que nenhuma camada está "desconectada" do grafo de
    gradientes (bug comum: esquecer de usar uma camada no forward())."""
    model = build_model(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()

    params_without_grad = [
        name for name, p in model.named_parameters() if p.grad is None
    ]
    assert not params_without_grad, f"Parâmetros sem gradiente: {params_without_grad}"


def test_overfit_single_batch():
    """O sanity check mais importante deste arquivo: um modelo saudável
    deve conseguir decorar 8 exemplos sintéticos em poucas iterações,
    levando a loss de ~2.3 (log(10), chute aleatório com 10 classes) a
    perto de 0. Se isso falhar, tem bug antes de gastar horas treinando
    no dataset real."""
    torch.manual_seed(0)
    model = build_model(num_classes=10)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    initial_loss = None
    final_loss = None
    for step in range(200):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss > 1.5, f"Loss inicial suspeita: {initial_loss} (esperado perto de ln(10)≈2.3)"
    assert final_loss < 0.1, f"Modelo não conseguiu overfit no batch pequeno (loss final={final_loss})"


if __name__ == "__main__":
    test_output_shape_matches_num_classes()
    test_works_with_different_batch_sizes()
    test_custom_num_classes()
    test_gradients_flow_through_all_parameters()
    test_overfit_single_batch()
    print("Todos os testes passaram!")
