import json, os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROC_DIR = "federated_data_proc"

# MUST match client/server model exactly
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
        return self.net(x).squeeze(1)

@torch.no_grad()
def main():
    with open(os.path.join(PROC_DIR, "meta.json")) as f:
        meta = json.load(f)

    input_dim = int(meta["input_dim"])
    model = MLP(input_dim)

    # For PyTorch 2.6+ safety defaults, this avoids odd issues:
    state = torch.load("model_out/global_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Combine all client validation sets
    Xs, ys = [], []
    for i in range(meta["n_clients"]):
        data = np.load(os.path.join(PROC_DIR, f"client_{i}.npz"))
        Xs.append(data["X_val"])
        ys.append(data["y_val"])

    X = torch.from_numpy(np.vstack(Xs)).float()
    y = torch.from_numpy(np.hstack(ys)).float()

    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False)

    correct, n = 0, 0
    for xb, yb in loader:
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()
        n += xb.size(0)

    print(f"🎯 Global model accuracy on combined val: {correct/n:.4f}")

if __name__ == "__main__":
    main()