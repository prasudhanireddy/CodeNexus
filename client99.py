import argparse, json, os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl

PROC_DIR = "federated_data_proc"
SERVER_ADDR = "127.0.0.1:8080"  # change if server is remote

class MLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits

def make_loaders(npz_path: str, batch_size: int):
    data = np.load(npz_path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)  # BCE expects float targets
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

def get_parameters(model):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_parameters(model, parameters):
    sd = model.state_dict()
    for (k, _), v in zip(sd.items(), parameters):
        sd[k] = torch.tensor(v)
    model.load_state_dict(sd, strict=True)

def train_one_epoch(model, loader, device, lr):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        bs = xb.size(0)
        total_loss += loss.item() * bs
        correct += (preds == yb).sum().item()
        n += bs

    return total_loss / max(n, 1), correct / max(n, 1)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, local_epochs, lr):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.train_loader, self.device, self.lr)
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(acc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--server", type=str, default=SERVER_ADDR)
    args = ap.parse_args()

    meta_path = os.path.join(PROC_DIR, "meta.json")
    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    input_dim = int(meta["input_dim"])
    npz_path = os.path.join(PROC_DIR, f"client_{args.cid}.npz")
    if not os.path.exists(npz_path):
        # In preprocess.py we saved files like client_0.npz because cid came from filename.
        # That is: client_0.csv -> client_0.npz
        # This path is correct. If not found, show directory contents.
        raise FileNotFoundError(f"Missing {npz_path}. Files in {PROC_DIR}: {os.listdir(PROC_DIR)}")

    train_loader, val_loader = make_loaders(npz_path, args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_features=input_dim)

    client = FlowerClient(model, train_loader, val_loader, device, args.local_epochs, args.lr)
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()