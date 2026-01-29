import os
import numpy as np
import pandas as pd

DATA_PATH = "TSAdataexercise_final1.xlsx"
OUT_DIR = "federated_data_raw"
LABEL_COL = "RETURNOR"     # change if needed
DROP_COLS = ["CASEID"]     # optional

N_CLIENTS = 3
SEED = 42

def stratified_partition_indices(y: np.ndarray, n_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)

    parts = [[] for _ in range(n_clients)]
    for k, idx in enumerate(idx0):
        parts[k % n_clients].append(idx)
    for k, idx in enumerate(idx1):
        parts[k % n_clients].append(idx)

    for c in range(n_clients):
        parts[c] = np.array(parts[c], dtype=np.int64)
        rng.shuffle(parts[c])
        if len(np.unique(y[parts[c]])) < 2:
            raise RuntimeError(f"Client {c} ended with one class. Try another SEED.")
    return parts

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_excel(DATA_PATH)

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if LABEL_COL not in df.columns:
        raise ValueError(f"'{LABEL_COL}' not found. Columns: {list(df.columns)}")

    y_raw = df[LABEL_COL].astype(str).str.strip().str.lower()
    y = y_raw.map({"yes": 1, "no": 0})
    if y.isna().any():
        bad = sorted(y_raw[y.isna()].unique().tolist())
        raise ValueError(f"Unrecognized values in {LABEL_COL}: {bad}")

    parts = stratified_partition_indices(y.to_numpy(), N_CLIENTS, SEED)

    for i, idx in enumerate(parts):
        client_df = df.iloc[idx].reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, f"client_{i}.csv")
        client_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} | n={len(client_df)} | pos_rate={(client_df[LABEL_COL].astype(str).str.lower()=='yes').mean():.3f}")

if __name__ == "__main__":
    main()