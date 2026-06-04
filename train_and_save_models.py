"""
train_and_save_models.py
------------------------
Run this script ONCE to train every model used by app.py and save the
results to the `models/` directory as pickle files.

Usage:
    python train_and_save_models.py

Size budget: every .pkl file must stay under 95 MiB.
Key levers to keep Random Forest small:
  - n_estimators=50  (fewer trees)
  - max_depth=12     (shallow trees -> exponentially fewer nodes)
  - min_samples_leaf=20  (prunes tiny leaves on 232k-row data)
  - max_features="sqrt"  (default, kept explicit for clarity)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/spotify_tracks.csv"
MODELS_DIR     = "models"
MAX_SIZE_BYTES = 95 * 1024 * 1024          # 95 MiB hard limit per file

os.makedirs(MODELS_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "tempo", "valence",
]
CLUSTER_FEATURES = ["danceability", "energy", "valence", "acousticness"]

# Random Forest hyperparameters tuned to keep pkl < 95 MiB
RF_PARAMS = dict(
    n_estimators=50,       # 50 trees is plenty for feature importance + prediction
    max_depth=12,          # shallow trees -> much smaller serialised size
    min_samples_leaf=20,   # prunes tiny leaves; robust on 232k rows
    max_features="sqrt",   # standard for regression forests
    random_state=42,
    n_jobs=-1,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(obj, name):
    """Pickle an object and assert it is within the size budget."""
    path = os.path.join(MODELS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_bytes = os.path.getsize(path)
    size_mib   = size_bytes / (1024 * 1024)
    print(f"  Saved -> {path}  ({size_mib:.2f} MiB)")
    if size_bytes > MAX_SIZE_BYTES:
        raise RuntimeError(
            f"ABORT: {name} is {size_mib:.1f} MiB, exceeds 95 MiB limit. "
            "Reduce n_estimators or max_depth and retrain."
        )

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()
print(f"  {len(df):,} rows loaded")

# ── Train regression models ───────────────────────────────────────────────────
print("Training regression models...")
X = df[NUMERIC_FEATURES]
y = df["popularity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(**RF_PARAMS)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

importance = pd.DataFrame({
    "Feature": NUMERIC_FEATURES,
    "Importance": rf.feature_importances_,
}).sort_values("Importance", ascending=False)

metrics = dict(
    lr=dict(
        mae=mean_absolute_error(y_test, lr_pred),
        mse=mean_squared_error(y_test, lr_pred),
        r2=r2_score(y_test, lr_pred),
    ),
    rf=dict(
        mae=mean_absolute_error(y_test, rf_pred),
        mse=mean_squared_error(y_test, rf_pred),
        r2=r2_score(y_test, rf_pred),
    ),
    importance=importance,
)

print(f"  LR  -> MAE={metrics['lr']['mae']:.2f}  R2={metrics['lr']['r2']:.3f}")
print(f"  RF  -> MAE={metrics['rf']['mae']:.2f}  R2={metrics['rf']['r2']:.3f}")

# ── Train full-dataset RF predictor ──────────────────────────────────────────
print("Training full-dataset Random Forest predictor...")
rf_full = RandomForestRegressor(**RF_PARAMS)
rf_full.fit(X, y)

# ── Train clustering models ───────────────────────────────────────────────────
print("Training K-Means + PCA clustering...")
scaler = StandardScaler()
scaled = scaler.fit_transform(df[CLUSTER_FEATURES])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled)
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(scaled)

clustering = dict(
    scaler=scaler,
    kmeans=kmeans,
    pca=pca,
    labels=labels,
    coords=coords,
    cluster_features=CLUSTER_FEATURES,
)

# ── Persist to disk ───────────────────────────────────────────────────────────
print("Saving models (limit: 95 MiB each)...")
save(lr,         "linear_regression.pkl")
save(rf,         "random_forest.pkl")
save(rf_full,    "random_forest_full.pkl")
save(metrics,    "regression_metrics.pkl")
save(clustering, "clustering.pkl")

print("\nDone! All models saved to the 'models/' directory.")
print("    You can now run `streamlit run app.py` -- no training will happen at startup.")
