import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Trend Analysis",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Overall dark background */
    .stApp { background-color: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Main text */
    body, .stMarkdown, p, li, label { color: #e6edf3 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a2236, #1c2a40);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"]  { color: #1db954 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"]  { color: #58a6ff !important; }

    /* Headers */
    h1 { color: #1db954 !important; font-size: 2.4rem !important; }
    h2 { color: #58a6ff !important; }
    h3 { color: #e6edf3 !important; }

    /* Section divider */
    hr { border-color: #30363d; }

    /* Sidebar headers */
    [data-testid="stSidebar"] h2 { color: #1db954 !important; }
    [data-testid="stSidebar"] h3 { color: #58a6ff !important; }
    [data-testid="stSidebar"] label { color: #e6edf3 !important; }

    /* Tabs / navbar */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 0;
        border-bottom: 2px solid #30363d;
        gap: 0;
        padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.4rem !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -2px !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e6edf3 !important;
        background: rgba(255,255,255,0.04) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1db954 !important;
        border-bottom: 2px solid #1db954 !important;
        background: transparent !important;
    }
    /* Pin the tab bar to the top of the main area */
    .stTabs { margin-top: -1rem; }

    /* DataFrames */
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="Inter, sans-serif"),
        title=dict(font=dict(color="#58a6ff", size=18)),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(t=60, b=50, l=60, r=30),
    )
)

SPOTIFY_GREEN = "#1db954"
SPOTIFY_BLUE  = "#58a6ff"
COLOR_SEQ     = px.colors.qualitative.Dark24

# ── Data loading & caching ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Spotify dataset…")
def load_data():
    df = pd.read_csv("data/spotify_tracks.csv")
    df = df.drop_duplicates()
    df["duration_min"] = df["duration_ms"] / 60_000
    df["popularity_category"] = pd.cut(
        df["popularity"],
        bins=[0, 35, 70, 100],
        labels=["Low", "Medium", "High"],
    )
    return df

# ── Feature / model caching ───────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "tempo", "valence",
]
MODELS_DIR = "models"

def _load_pkl(filename):
    """Load a pickle file from the models directory."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading regression models…")
def load_regression_models():
    """Load pre-trained regression models and metrics from disk."""
    metrics     = _load_pkl("regression_metrics.pkl")
    hgb_model   = _load_pkl("hgb_full.pkl")
    feature_meta = _load_pkl("feature_meta.pkl")
    return metrics, hgb_model, feature_meta

@st.cache_resource(show_spinner="Loading clustering models…")
def load_clustering_models():
    """Load pre-trained clustering artefacts from disk."""
    return _load_pkl("clustering.pkl")

def _models_exist():
    required = [
        "regression_metrics.pkl",
        "hgb_full.pkl",
        "feature_meta.pkl",
        "clustering.pkl",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)

# ── Helper: apply template to a figure ───────────────────────────────────────
def styled(fig, title="", height=420):
    fig.update_layout(
        **PLOT_TEMPLATE["layout"],
        title_text=title,
        height=height,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
df_all = load_data()

with st.sidebar:
    st.markdown("## Spotify Trends")
    st.markdown("---")

    st.markdown("### Filters")
    all_genres = sorted(df_all["genre"].unique().tolist())
    selected_genres = st.multiselect(
        "Genres",
        options=all_genres,
        default=all_genres[:10],
        help="Select genres to include in analysis",
    )

    pop_range = st.slider(
        "Popularity range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=1,
    )



# ── Filter dataframe ──────────────────────────────────────────────────────────
if selected_genres:
    df = df_all[
        df_all["genre"].isin(selected_genres) &
        df_all["popularity"].between(pop_range[0], pop_range[1])
    ].copy()
else:
    df = df_all[df_all["popularity"].between(pop_range[0], pop_range[1])].copy()

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR (horizontal tabs)
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_pop, tab_audio, tab_ml, tab_cluster = st.tabs(
    ["Overview", "Popularity", "Audio Features", "ML Prediction", "Clustering"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.title("Spotify Trend Analysis")
    st.markdown("Explore **232k+ Spotify tracks** — what makes songs popular, how features correlate, and what clusters emerge.")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tracks", f"{len(df):,}")
    c2.metric("Genres", f"{df['genre'].nunique():,}")
    c3.metric("Artists", f"{df['artist_name'].nunique():,}")
    c4.metric("Avg Popularity", f"{df['popularity'].mean():.1f} / 100")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Sample Records")
        st.dataframe(
            df[["genre", "artist_name", "track_name", "popularity",
                "danceability", "energy", "tempo", "valence"]].head(10),
            width='stretch',
        )

    with col_r:
        st.subheader("Descriptive Statistics")
        desc = df[["popularity"] + NUMERIC_FEATURES].describe().round(3)
        st.dataframe(desc, width='stretch')

    st.markdown("---")

    # Genre distribution
    genre_counts = df["genre"].value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]
    fig = px.bar(
        genre_counts.head(20), x="count", y="genre",
        orientation="h", color="count",
        color_continuous_scale=[[0, "#1a2a1f"], [1, SPOTIFY_GREEN]],
        labels={"count": "Track count", "genre": ""},
    )
    fig = styled(fig, "Top 20 Genres by Track Count")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB: POPULARITY
# ─────────────────────────────────────────────────────────────────────────────
with tab_pop:
    st.title("Popularity Analysis")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(
            df, x="popularity", nbins=50,
            color_discrete_sequence=[SPOTIFY_GREEN],
            labels={"popularity": "Popularity Score", "count": "Tracks"},
        )
        fig = styled(fig, "Distribution of Popularity")
        st.plotly_chart(fig, width='stretch')

    with col_r:
        cat_counts = df["popularity_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.pie(
            cat_counts, names="category", values="count",
            color_discrete_sequence=[SPOTIFY_GREEN, SPOTIFY_BLUE, "#ff7b54"],
            hole=0.4,
        )
        fig = styled(fig, "Popularity Category Split")
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Average Popularity by Genre")

    genre_pop = (
        df.groupby("genre")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .head(25)
    )
    fig = px.bar(
        genre_pop, x="genre", y="popularity",
        color="popularity",
        color_continuous_scale=[[0, "#1a2a1f"], [1, SPOTIFY_GREEN]],
        labels={"popularity": "Avg Popularity", "genre": ""},
    )
    fig = styled(fig, "", height=450)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=-40)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Popularity vs Duration")
    fig = px.scatter(
        df.sample(min(5000, len(df)), random_state=42),
        x="duration_min", y="popularity",
        opacity=0.35,
        color_discrete_sequence=[SPOTIFY_GREEN],
        labels={"duration_min": "Duration (min)", "popularity": "Popularity"},
        trendline="ols",
        trendline_color_override=SPOTIFY_BLUE,
    )
    fig = styled(fig, "Popularity vs Track Duration", height=430)
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB: AUDIO FEATURES
# ─────────────────────────────────────────────────────────────────────────────
with tab_audio:
    st.title("Audio Feature Explorer")
    st.markdown("---")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_cols = ["popularity"] + NUMERIC_FEATURES + ["duration_min"]
    corr = df[corr_cols].corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[
                [0.0, "#d62728"], [0.5, "#161b22"], [1.0, SPOTIFY_GREEN]
            ],
            zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 11},
            hoverongaps=False,
        )
    )
    fig = styled(fig, "Feature Correlation Matrix", height=520)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Feature vs popularity scatter
    st.subheader("Audio Feature vs Popularity")
    feat = st.selectbox("Select feature", NUMERIC_FEATURES, index=0)
    sample_df = df.sample(min(8000, len(df)), random_state=42)

    fig = px.scatter(
        sample_df, x=feat, y="popularity",
        opacity=0.3, color_discrete_sequence=[SPOTIFY_GREEN],
        labels={feat: feat.replace("_", " ").title(), "popularity": "Popularity"},
        trendline="ols",
        trendline_color_override=SPOTIFY_BLUE,
    )
    fig = styled(fig, f"{feat.replace('_',' ').title()} vs Popularity", height=430)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Box-plots by category
    st.subheader("Feature Distribution by Popularity Category")
    feat2 = st.selectbox("Select feature for box plot", NUMERIC_FEATURES, index=1, key="box_feat")
    fig = px.box(
        df, x="popularity_category", y=feat2,
        color="popularity_category",
        color_discrete_sequence=[SPOTIFY_GREEN, SPOTIFY_BLUE, "#ff7b54"],
        category_orders={"popularity_category": ["Low", "Medium", "High"]},
        labels={"popularity_category": "Popularity Category",
                feat2: feat2.replace("_", " ").title()},
    )
    fig = styled(fig, f"{feat2.replace('_',' ').title()} by Popularity Category", height=430)
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB: ML PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_ml:
    st.title("Popularity Prediction (ML)")
    st.markdown("Models are loaded from pre-trained **`.pkl` files** — no training happens at runtime.")
    st.markdown("---")

    if not _models_exist():
        st.error(
            "⚠️ Pre-trained model files not found in the `models/` directory.\n\n"
            "Please run the training script first:\n```\npython train_and_save_models.py\n```"
        )
    else:
        results, hgb_model, feature_meta = load_regression_models()

        col_l, col_m, col_r = st.columns(3)

        with col_l:
            st.subheader("Linear Regression")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",  f"{results['lr']['mae']:.2f}")
            m2.metric("RMSE", f"{results['lr']['mse']**0.5:.2f}")
            m3.metric("R²",   f"{results['lr']['r2']:.3f}")

        with col_m:
            st.subheader("Random Forest")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",  f"{results['rf']['mae']:.2f}")
            m2.metric("RMSE", f"{results['rf']['mse']**0.5:.2f}")
            m3.metric("R²",   f"{results['rf']['r2']:.3f}")

        with col_r:
            st.subheader("Gradient Boosting")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",  f"{results['hgb']['mae']:.2f}")
            m2.metric("RMSE", f"{results['hgb']['mse']**0.5:.2f}")
            m3.metric("R²",   f"{results['hgb']['r2']:.3f}")

        st.markdown("---")

        st.subheader("Feature Importance (Grouped)")
        imp = results.get("importance_grouped", results["importance"])
        # Show top 15 features for readability
        imp_display = imp.head(15)
        fig = px.bar(
            imp_display, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#1a2a1f"], [1, SPOTIFY_GREEN]],
            labels={"Importance": "Importance Score", "Feature": ""},
        )
        fig = styled(fig, "Feature Importance (categories grouped)", height=480)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # Interactive predictor
        st.subheader("Predict Popularity for a Custom Track")
        st.markdown("Adjust the inputs below and see how the **Gradient Boosting** model would rate a hypothetical track.")

        # Categorical inputs
        cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
        with cat_col1:
            pred_genre = st.selectbox("Genre", sorted(df_all["genre"].unique().tolist()), index=0)
        with cat_col2:
            pred_key = st.selectbox("Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], index=0)
        with cat_col3:
            pred_mode = st.selectbox("Mode", ["Major", "Minor"], index=0)
        with cat_col4:
            pred_timesig = st.selectbox("Time Signature", ["4/4", "3/4", "5/4", "1/4", "0/4"], index=0)

        col1, col2, col3 = st.columns(3)
        with col1:
            dance = st.slider("Danceability", 0.0, 1.0, 0.55, 0.01)
            energy = st.slider("Energy", 0.0, 1.0, 0.57, 0.01)
            loudness = st.slider("Loudness (dB)", -52.0, 4.0, -9.5, 0.5)
            duration_ms = st.slider("Duration (sec)", 15, 600, 210, 5) * 1000
        with col2:
            speech = st.slider("Speechiness", 0.0, 1.0, 0.12, 0.01)
            acoustic = st.slider("Acousticness", 0.0, 1.0, 0.37, 0.01)
            instrumental = st.slider("Instrumentalness", 0.0, 1.0, 0.15, 0.01)
        with col3:
            liveness = st.slider("Liveness", 0.0, 1.0, 0.22, 0.01)
            tempo = st.slider("Tempo (BPM)", 30.0, 243.0, 118.0, 1.0)
            valence = st.slider("Valence", 0.0, 1.0, 0.45, 0.01)

        # Build input DataFrame matching the training feature set
        custom_data = {
            "danceability": [dance], "energy": [energy], "loudness": [loudness],
            "speechiness": [speech], "acousticness": [acoustic],
            "instrumentalness": [instrumental], "liveness": [liveness],
            "tempo": [tempo], "valence": [valence], "duration_ms": [duration_ms],
            "energy_x_dance": [energy * dance],
            "loud_x_energy": [loudness * energy],
            "acoustic_inverse": [1 - acoustic],
            "genre": [pred_genre], "key": [pred_key],
            "mode": [pred_mode], "time_signature": [pred_timesig],
        }
        custom_df = pd.DataFrame(custom_data)
        # Use the feature column order from training
        custom_df = custom_df[feature_meta["all_feature_cols"]]
        pred = hgb_model.predict(custom_df)[0]

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a2236, #1c2a40);
            border: 1px solid #1db954;
            border-radius: 14px;
            padding: 1.5rem 2rem;
            margin-top: 1rem;
            text-align: center;
        ">
            <p style="color:#8b949e; margin:0; font-size:0.9rem;">Predicted Popularity (Gradient Boosting)</p>
            <p style="color:#1db954; font-size:3rem; font-weight:800; margin:0.2rem 0;">{pred:.1f}<span style="font-size:1.5rem; color:#8b949e"> / 100</span></p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
with tab_cluster:
    st.title("Song Clustering (K-Means)")
    st.markdown("Songs are grouped into **3 clusters** based on danceability, energy, valence & acousticness.")
    st.markdown("---")

    _clustering = load_clustering_models()
    if _clustering is None:
        st.error(
            "⚠️ Clustering model not found in the `models/` directory.\n\n"
            "Please run the training script first:\n```\npython train_and_save_models.py\n```"
        )
        st.stop()

    labels       = _clustering["labels"]
    coords       = _clustering["coords"]
    cluster_feats = _clustering["cluster_features"]

    cluster_df = df_all.copy()
    cluster_df["Cluster"] = labels.astype(str)
    cluster_df["PCA_1"] = coords[:, 0]
    cluster_df["PCA_2"] = coords[:, 1]

    # PCA scatter (sample for performance)
    sample = cluster_df.sample(min(8000, len(cluster_df)), random_state=42)
    fig = px.scatter(
        sample, x="PCA_1", y="PCA_2",
        color="Cluster",
        color_discrete_sequence=[SPOTIFY_GREEN, SPOTIFY_BLUE, "#ff7b54"],
        opacity=0.45,
        hover_data=["artist_name", "track_name", "genre"],
        labels={"PCA_1": "PCA Component 1", "PCA_2": "PCA Component 2"},
    )
    fig = styled(fig, "K-Means Clusters (PCA 2D Projection)", height=500)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Cluster Profiles")

    cluster_profile = (
        cluster_df.groupby("Cluster")[cluster_feats + ["popularity"]]
        .mean()
        .round(3)
        .reset_index()
    )
    cluster_profile.columns = ["Cluster"] + [c.replace("_", " ").title() for c in cluster_profile.columns[1:]]

    # Radar chart per cluster
    cats = [c.replace("_", " ").title() for c in cluster_feats]
    colors = [SPOTIFY_GREEN, SPOTIFY_BLUE, "#ff7b54"]
    fig_radar = go.Figure()
    for i, row in cluster_profile.iterrows():
        vals = [row[c] for c in cats]
        vals += [vals[0]]  # close polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats + [cats[0]],
            fill="toself",
            name=f"Cluster {row['Cluster']}",
            line_color=colors[i],
            fillcolor=colors[i],
            opacity=0.3,
        ))
    fig_radar.update_layout(
        **PLOT_TEMPLATE["layout"],
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            angularaxis=dict(gridcolor="#21262d", tickfont=dict(color="#e6edf3")),
        ),
        title_text="Cluster Audio Feature Profiles",
        height=450,
    )
    st.plotly_chart(fig_radar, width='stretch')

    st.markdown("---")
    st.subheader("Cluster Summary Table")
    st.dataframe(cluster_profile, width='stretch')

    st.markdown("---")
    st.subheader("Cluster Size Distribution")
    size_df = cluster_df["Cluster"].value_counts().reset_index()
    size_df.columns = ["Cluster", "Count"]
    fig = px.bar(
        size_df, x="Cluster", y="Count",
        color="Cluster",
        color_discrete_sequence=[SPOTIFY_GREEN, SPOTIFY_BLUE, "#ff7b54"],
        labels={"Count": "Number of Tracks", "Cluster": "Cluster"},
    )
    fig = styled(fig, "Tracks per Cluster", height=380)
    st.plotly_chart(fig, width='stretch')
