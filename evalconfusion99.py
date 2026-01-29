import json
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

PROC_DIR = "federated_data_proc"
MODEL_PATH = "model_out/global_model.pt"


# MUST MATCH client/server model EXACTLY
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
    # Load metadata
    with open(os.path.join(PROC_DIR, "meta.json"), "r") as f:
        meta = json.load(f)

    input_dim = int(meta["input_dim"])
    n_clients = int(meta["n_clients"])

    # Load model
    model = MLP(input_dim)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Collect all validation data from clients
    X_all, y_all = [], []
    for i in range(n_clients):
        data = np.load(os.path.join(PROC_DIR, f"client_{i}.npz"))
        X_all.append(data["X_val"])
        y_all.append(data["y_val"])

    X = torch.from_numpy(np.vstack(X_all)).float()
    y = torch.from_numpy(np.hstack(y_all)).float()

    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False)

    y_true, y_pred = [], []

    for xb, yb in loader:
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()

        y_true.extend(yb.int().tolist())
        y_pred.extend(preds.tolist())

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Confusion Matrix (Global Model)")
    print("Predicted →")
    print("           0     1")
    print(f"Actual 0 | {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"Actual 1 | {cm[1,0]:5d} {cm[1,1]:5d}")

    # Classification report
    print("\n📈 Classification Report")
    print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))


if __name__ == "__main__":
    main()