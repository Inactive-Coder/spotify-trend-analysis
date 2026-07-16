"""
train_and_save_models.py
------------------------
Run this script ONCE to train every model used by app.py and save the
results to the `models/` directory as pickle files.

Usage:
    python train_and_save_models.py

Size budget: every .pkl file must stay under 95 MiB.

Models trained:
  - Linear Regression (baseline)
  - Random Forest Regressor (improved hyperparameters)
  - HistGradient Boosting Regressor (best accuracy)
  - K-Means + PCA clustering
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/spotify_tracks.csv"
MODELS_DIR     = "models"
MAX_SIZE_BYTES = 95 * 1024 * 1024          # 95 MiB hard limit per file

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature definitions ──────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "tempo", "valence",
    "duration_ms",
]

CATEGORICAL_FEATURES = ["genre", "key", "mode", "time_signature"]

ENGINEERED_FEATURES = ["energy_x_dance", "loud_x_energy", "acoustic_inverse"]

CLUSTER_FEATURES = ["danceability", "energy", "valence", "acousticness"]

# Random Forest hyperparameters (relaxed for better accuracy)
RF_PARAMS = dict(
    n_estimators=100,      # increased from 50
    max_depth=18,          # increased from 12
    min_samples_leaf=10,   # decreased from 20
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

# HistGradientBoosting hyperparameters
HGB_PARAMS = dict(
    max_iter=300,
    max_depth=8,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42,
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


def engineer_features(dataframe):
    """Add engineered features to a dataframe (in-place)."""
    dataframe["energy_x_dance"]   = dataframe["energy"] * dataframe["danceability"]
    dataframe["loud_x_energy"]    = dataframe["loudness"] * dataframe["energy"]
    dataframe["acoustic_inverse"] = 1 - dataframe["acousticness"]
    return dataframe


def build_preprocessor():
    """Build a ColumnTransformer for numeric + categorical features."""
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()
print(f"  {len(df):,} rows loaded")

# ── Engineer features ─────────────────────────────────────────────────────────
print("Engineering features...")
df = engineer_features(df)

# ── Prepare features and target ──────────────────────────────────────────────
all_feature_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
X = df[all_feature_cols]
y = df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Build preprocessing pipeline ─────────────────────────────────────────────
preprocessor = build_preprocessor()

# ── Train Linear Regression ──────────────────────────────────────────────────
print("\nTraining Linear Regression...")
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
print(f"  LR  -> MAE={mean_absolute_error(y_test, lr_pred):.2f}  "
      f"R2={r2_score(y_test, lr_pred):.3f}")

# ── Train Random Forest ──────────────────────────────────────────────────────
print("\nTraining Random Forest (relaxed hyperparameters)...")
rf_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", RandomForestRegressor(**RF_PARAMS)),
])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
print(f"  RF  -> MAE={mean_absolute_error(y_test, rf_pred):.2f}  "
      f"R2={r2_score(y_test, rf_pred):.3f}")

# ── Train HistGradientBoosting ────────────────────────────────────────────────
print("\nTraining HistGradient Boosting Regressor...")
hgb_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", HistGradientBoostingRegressor(**HGB_PARAMS)),
])
hgb_pipeline.fit(X_train, y_train)
hgb_pred = hgb_pipeline.predict(X_test)
print(f"  HGB -> MAE={mean_absolute_error(y_test, hgb_pred):.2f}  "
      f"R2={r2_score(y_test, hgb_pred):.3f}")

# ── Compute feature importance (from RF) ─────────────────────────────────────
# Get feature names from the preprocessor after fitting
rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
cat_encoder = rf_preprocessor.named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
all_transformed_names = (NUMERIC_FEATURES + ENGINEERED_FEATURES + cat_feature_names)

rf_importances = rf_pipeline.named_steps["model"].feature_importances_
importance = pd.DataFrame({
    "Feature": all_transformed_names,
    "Importance": rf_importances,
}).sort_values("Importance", ascending=False)

# Also create a grouped importance (sum one-hot categories back together)
def group_importance(importance_df):
    """Group one-hot encoded feature importances back to original categories."""
    grouped = {}
    for _, row in importance_df.iterrows():
        feat = row["Feature"]
        imp = row["Importance"]
        # Check if it's a one-hot encoded feature (contains underscore from OHE)
        matched = False
        for cat_feat in CATEGORICAL_FEATURES:
            if feat.startswith(cat_feat + "_"):
                grouped[cat_feat] = grouped.get(cat_feat, 0) + imp
                matched = True
                break
        if not matched:
            grouped[feat] = imp
    return pd.DataFrame({
        "Feature": list(grouped.keys()),
        "Importance": list(grouped.values()),
    }).sort_values("Importance", ascending=False)

importance_grouped = group_importance(importance)

# ── Collect metrics ──────────────────────────────────────────────────────────
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
    hgb=dict(
        mae=mean_absolute_error(y_test, hgb_pred),
        mse=mean_squared_error(y_test, hgb_pred),
        r2=r2_score(y_test, hgb_pred),
    ),
    importance=importance,
    importance_grouped=importance_grouped,
)

print(f"\n{'='*60}")
print(f"  Model Comparison Summary")
print(f"{'='*60}")
print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
for name, key in [("Linear Regression", "lr"), ("Random Forest", "rf"),
                   ("Hist Gradient Boost", "hgb")]:
    m = metrics[key]
    print(f"  {name:<25} {m['mae']:>8.2f} {m['mse']**0.5:>8.2f} {m['r2']:>8.3f}")
print(f"{'='*60}")

# ── Train full-dataset best model (HGB) for prediction ──────────────────────
print("\nTraining full-dataset HistGradient Boosting predictor...")
hgb_full_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", HistGradientBoostingRegressor(**HGB_PARAMS)),
])
hgb_full_pipeline.fit(X, y)

# ── Also train full-dataset RF for comparison ────────────────────────────────
print("Training full-dataset Random Forest predictor...")
rf_full_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", RandomForestRegressor(**RF_PARAMS)),
])
rf_full_pipeline.fit(X, y)

# ── Feature metadata (for app.py to reconstruct features at predict time) ────
feature_meta = dict(
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    engineered_features=ENGINEERED_FEATURES,
    all_feature_cols=all_feature_cols,
)

# ── Train clustering models ───────────────────────────────────────────────────
print("\nTraining K-Means + PCA clustering...")
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
print("\nSaving models (limit: 95 MiB each)...")
save(lr_pipeline,       "linear_regression.pkl")
save(rf_pipeline,       "random_forest.pkl")
save(hgb_pipeline,      "hist_gradient_boosting.pkl")
save(rf_full_pipeline,  "random_forest_full.pkl")
save(hgb_full_pipeline, "hgb_full.pkl")
save(metrics,           "regression_metrics.pkl")
save(feature_meta,      "feature_meta.pkl")
save(clustering,        "clustering.pkl")

print("\nDone! All models saved to the 'models/' directory.")
print("    You can now run `streamlit run app.py` -- no training happens at startup.")
