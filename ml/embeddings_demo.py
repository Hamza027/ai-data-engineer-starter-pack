from __future__ import annotations

import torch


def make_data(n: int = 2000, seed: int = 42):
    torch.manual_seed(seed)
    x = torch.rand(n, 2) * 2 - 1
    r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    y = (r > 0.6).long()
    return x, y


class EmbedNet(torch.nn.Module):
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.embed = torch.nn.Linear(16, embed_dim)
        self.head = torch.nn.Linear(embed_dim, 2)

        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        e = self.embed(x)          # embedding
        logits = self.head(self.act(e))
        return logits, e


def cosine_sim(query: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    # query: (d,) , mat: (n,d)
    q = query / (query.norm() + 1e-12)
    m = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    return (m @ q)


def main() -> None:
    x, y = make_data(n=3000)
    x_train, y_train = x[:2500], y[:2500]
    x_val, y_val = x[2500:], y[2500:]

    model = EmbedNet(embed_dim=8)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # train quickly
    for epoch in range(1, 11):
        idx = torch.randperm(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

        model.train()
        for i in range(0, len(x_train), 128):
            xb = x_train[i : i + 128]
            yb = y_train[i : i + 128]
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_logits, val_emb = model(x_val)
        preds = torch.argmax(val_logits, dim=1)
        acc = float((preds == y_val).float().mean().item())
        print(f"val_acc={acc:.3f}  embeddings_shape={tuple(val_emb.shape)}")

        # pick one query point from validation set
        q_idx = 0
        q_point = x_val[q_idx]
        q_label = int(y_val[q_idx].item())
        q_embed = val_emb[q_idx]

        sims = cosine_sim(q_embed, val_emb)
        topk = torch.topk(sims, k=6)  # includes itself

        print("\nQuery point:", q_point.tolist(), "label=", q_label)
        print("Top similar points (by cosine on embeddings):")

        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            pt = x_val[idx].tolist()
            lbl = int(y_val[idx].item())
            print(f"cos={score:.4f}  label={lbl}  point={pt}")


if __name__ == "__main__":
    main()
