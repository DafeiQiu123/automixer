# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


def train_model(model,
                dataset,
                epochs: int = 10,
                batch_size: int = 4,
                lr: float = 1e-4,
                device: str | None = None):
    """
    Simple L1 regression training for DSP parameters.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, Y in tqdm(loader, desc=f"Epoch {epoch}"):
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch}] Avg L1 Loss: {avg_loss:.4f}")
