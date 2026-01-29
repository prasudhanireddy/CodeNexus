import os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RAW_DIR = "federated_data_raw"
OUT_DIR = "federated_data_proc"

LABEL_COL = "RETURNOR"   # change if needed
SEED = 42

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols),
                      ("cat", categorical_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )

def encode_label(y_series: pd.Series) -> np.ndarray:
    y_raw = y_series.astype(str).str.strip().str.lower()
    y = y_raw.map({"yes": 1, "no": 0})
    if y.isna().any():
        bad = sorted(y_raw[y.isna()].unique().tolist())
        raise ValueError(f"Unrecognized label values in '{LABEL_COL}': {bad}")
    return y.to_numpy(dtype=np.int64)

def densify(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all client CSVs
    client_files = sorted([f for f in os.listdir(RAW_DIR) if f.startswith("client_") and f.endswith(".csv")])
    if not client_files:
        raise FileNotFoundError(f"No client_*.csv found in {RAW_DIR}. Run split_data.py first.")

    clients = []
    for f in client_files:
        df = pd.read_csv(os.path.join(RAW_DIR, f))
        if LABEL_COL not in df.columns:
            raise ValueError(f"'{LABEL_COL}' not found in {f}")
        X = df.drop(columns=[LABEL_COL])
        y = encode_label(df[LABEL_COL])
        clients.append((f, X, y))

    # Fit preprocessor globally on ALL clients' features
    X_all = pd.concat([X for _, X, _ in clients], axis=0, ignore_index=True)
    pre = build_preprocessor(X_all)
    pre.fit(X_all)

    joblib.dump(pre, os.path.join(OUT_DIR, "preprocessor.joblib"))

    # Transform each client and split into train/val
    input_dim = None
    for f, X, y in clients:
        Xt = densify(pre.transform(X))

        X_train, X_val, y_train, y_val = train_test_split(
            Xt, y, test_size=0.2, random_state=SEED, stratify=y
        )

        if input_dim is None:
            input_dim = Xt.shape[1]

        cid = f.replace(".csv", "")
        np.savez_compressed(
            os.path.join(OUT_DIR, f"{cid}.npz"),
            X_train=X_train, y_train=y_train.astype(np.int64),
            X_val=X_val, y_val=y_val.astype(np.int64),
        )
        print(f"Saved {OUT_DIR}/{cid}.npz | X shape={Xt.shape} | pos_rate={y.mean():.3f}")

    meta = {"label_col": LABEL_COL, "n_clients": len(clients), "input_dim": int(input_dim), "seed": SEED}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    print("Saved meta.json")

if __name__ == "__main__":
    main()