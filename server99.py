import os
import json
import flwr as fl
import torch
from torch import nn
from flwr.common import parameters_to_ndarrays  # <-- key fix

PROC_DIR = "federated_data_proc"
SERVER_ADDRESS = "0.0.0.0:8080"
NUM_ROUNDS = 10


# Must match client architecture exactly
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


def set_parameters_from_ndarrays(model: nn.Module, ndarrays):
    """Load list of numpy arrays into a PyTorch model state_dict."""
    state_dict = model.state_dict()
    for (k, _), v in zip(state_dict.items(), ndarrays):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        # aggregated is usually: (Parameters, metrics) or None
        if aggregated is None:
            return None

        params, metrics = aggregated

        # Save at the final round
        if server_round == NUM_ROUNDS:
            # ✅ Convert Parameters -> list[np.ndarray]
            ndarrays = parameters_to_ndarrays(params)

            with open(os.path.join(PROC_DIR, "meta.json"), "r") as f:
                meta = json.load(f)

            model = MLP(meta["input_dim"])
            set_parameters_from_ndarrays(model, ndarrays)

            os.makedirs("model_out", exist_ok=True)
            torch.save(model.state_dict(), "model_out/global_model.pt")
            print("\n✅ Saved global model to model_out/global_model.pt\n")

        return params, metrics


def main():
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()