from __future__ import annotations

import math
import torch


def make_data(n: int = 2000, seed: int = 42):
    torch.manual_seed(seed)
    x = torch.rand(n, 2) * 2 - 1  # uniform in [-1, 1] for each dim
    r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    y = (r > 0.6).long()  # 1 if outside circle radius 0.6 else 0
    return x, y


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),  # 2 classes: 0 or 1
        )

    def forward(self, x):
        return self.net(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def main() -> None:
    x, y = make_data(n=3000)
    x_train, y_train = x[:2500], y[:2500]
    x_val, y_val = x[2500:], y[2500:]

    model = TinyNet()
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 128
    epochs = 20

    for epoch in range(1, epochs + 1):
        # shuffle each epoch
        idx = torch.randperm(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

        model.train()
        total_loss = 0.0

        for i in range(0, len(x_train), batch_size):
            xb = x_train[i : i + batch_size]
            yb = y_train[i : i + batch_size]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_acc = accuracy(val_logits, y_val)

        if epoch in [1, 2, 3, 5, 10, 15, 20]:
            print(
                f"epoch={epoch:2d}  train_loss={total_loss:.3f}  val_loss={val_loss:.3f}  val_acc={val_acc:.3f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
