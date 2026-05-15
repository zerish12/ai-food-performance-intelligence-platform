
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, r2_score


st.set_page_config(
    page_title="Food Performance Intelligence Platform",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐶 AI-Powered Food Performance & Recommendation Intelligence Platform")
st.markdown("""
**Developed by Dr Muhammad Zahir Khan**  
Applied Statistics | Product Analytics | Machine Learning | Decision Science
""")
st.caption(
    "Product analytics, behavioural analytics, QC intelligence, retention modelling, "
    "experimentation, and recommendation optimisation."
)

db_path = "food_performance.db"
st.sidebar.header("Data Connection")

db_path = st.sidebar.text_input(
    "SQLite database path",
    value=DEFAULT_LOCAL_PATH
)

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload analytical_dataset.csv",
    type=["csv"]
)


@st.cache_data(show_spinner=True)
def load_from_csv(file):
    return pd.read_csv(file)


@st.cache_data(show_spinner=True)
def load_from_sqlite(path):
    conn = sqlite3.connect(path)
    data = pd.read_sql_query("SELECT * FROM analytical_dataset", conn)
    conn.close()
    return data


def load_data():
    if uploaded_file is not None:
        return load_from_csv(uploaded_file)

    path = Path(db_path)

    if not path.exists():
        st.error("Database not found. Check the path or upload analytical_dataset.csv.")
        st.stop()

    return load_from_sqlite(str(path))


df = load_data()


def prepare_features(data):
    data = data.copy()

    data["high_nir_anomaly"] = (data["nir_anomaly_score"] >= 2.0).astype(int)
    data["new_recommendation"] = (data["recommendation_version"] == "New").astype(int)
    data["senior_dog"] = (data["dog_age_years"] >= 8).astype(int)
    data["puppy"] = (data["dog_age_years"] < 1.5).astype(int)
    data["low_satisfaction"] = (data["satisfaction_score"] < 3.5).astype(int)
    data["high_satisfaction"] = (data["satisfaction_score"] >= 4.0).astype(int)

    raw_score = (
        0.35 * data["satisfaction_score"]
        + 0.20 * data["repeat_order"]
        + 0.10 * data["new_recommendation"]
        - 0.25 * data["cancelled"]
        - 0.15 * data["flavour_switched"]
        - 0.15 * data["complaint"]
        - 0.10 * data["high_nir_anomaly"]
    )

    scaler = MinMaxScaler(feature_range=(0, 100))
    data["food_performance_score"] = scaler.fit_transform(raw_score.values.reshape(-1, 1))

    return data


df = prepare_features(df)


st.sidebar.header("Interactive Filters")

recipe_filter = st.sidebar.multiselect(
    "Recipe Type",
    sorted(df["recipe_type"].dropna().unique()),
    default=sorted(df["recipe_type"].dropna().unique())
)

dog_size_filter = st.sidebar.multiselect(
    "Dog Size",
    sorted(df["dog_size"].dropna().unique()),
    default=sorted(df["dog_size"].dropna().unique())
)

recommendation_filter = st.sidebar.multiselect(
    "Recommendation Version",
    sorted(df["recommendation_version"].dropna().unique()),
    default=sorted(df["recommendation_version"].dropna().unique())
)

sensitive_filter = st.sidebar.multiselect(
    "Sensitive Stomach",
    sorted(df["sensitive_stomach"].dropna().unique()),
    default=sorted(df["sensitive_stomach"].dropna().unique())
)

filtered = df[
    (df["recipe_type"].isin(recipe_filter))
    & (df["dog_size"].isin(dog_size_filter))
    & (df["recommendation_version"].isin(recommendation_filter))
    & (df["sensitive_stomach"].isin(sensitive_filter))
].copy()

if filtered.empty:
    st.warning("No data after filters. Please adjust selections.")
    st.stop()


@st.cache_resource(show_spinner=True)
def train_churn_model(data):
    features = [
        "recommendation_version", "satisfaction_score", "flavour_switched",
        "repeat_order", "complaint", "dog_size", "dog_age_years",
        "activity_level", "sensitive_stomach", "breed_group", "recipe_type",
        "nir_anomaly_score", "moisture_deviation", "protein_deviation",
        "fat_deviation", "qc_pass"
    ]

    target = "cancelled"
    model_df = data[features + [target]].dropna()

    X = model_df[features]
    y = model_df[target]

    categorical = [
        "recommendation_version", "dog_size", "activity_level",
        "breed_group", "recipe_type"
    ]

    numeric = [
        "satisfaction_score", "flavour_switched", "repeat_order",
        "complaint", "dog_age_years", "sensitive_stomach",
        "nir_anomaly_score", "moisture_deviation", "protein_deviation",
        "fat_deviation", "qc_pass"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=250,
                max_depth=7,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

    return model, metrics, features


@st.cache_resource(show_spinner=True)
def train_score_model(data):
    features = [
        "dog_size", "dog_age_years", "activity_level", "sensitive_stomach",
        "breed_group", "recipe_type", "protein_pct", "fat_pct", "fibre_pct",
        "nir_anomaly_score", "moisture_deviation", "protein_deviation",
        "fat_deviation", "qc_pass"
    ]

    target = "food_performance_score"
    model_df = data[features + [target]].dropna()

    X = model_df[features]
    y = model_df[target]

    categorical = ["dog_size", "activity_level", "breed_group", "recipe_type"]

    numeric = [
        "dog_age_years", "sensitive_stomach", "protein_pct", "fat_pct",
        "fibre_pct", "nir_anomaly_score", "moisture_deviation",
        "protein_deviation", "fat_deviation", "qc_pass"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                random_state=42
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    return model, metrics, features


churn_model, churn_metrics, churn_features = train_churn_model(df)
score_model, score_metrics, score_features = train_score_model(df)


tabs = st.tabs([
    "Executive Overview",
    "Food Performance",
    "Behaviour & Retention",
    "Experimentation",
    "QC / NIR Intelligence",
    "NLP Feedback",
    "Recommendation Engine",
    "Scenario Simulator",
    "Model Performance"
])


with tabs[0]:
    st.subheader("Executive Overview")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Customers", f"{len(filtered):,}")
    c2.metric("Cancellation Rate", f"{filtered['cancelled'].mean():.1%}")
    c3.metric("Repeat Rate", f"{filtered['repeat_order'].mean():.1%}")
    c4.metric("Switch Rate", f"{filtered['flavour_switched'].mean():.1%}")
    c5.metric("Avg Satisfaction", f"{filtered['satisfaction_score'].mean():.2f}")
    c6.metric("Food Score", f"{filtered['food_performance_score'].mean():.1f}")

    recipe_summary = (
        filtered.groupby("recipe_type")
        .agg(
            customers=("customer_id", "count"),
            cancellation_rate=("cancelled", "mean"),
            repeat_rate=("repeat_order", "mean"),
            satisfaction=("satisfaction_score", "mean"),
            food_score=("food_performance_score", "mean")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            recipe_summary.sort_values("food_score"),
            x="food_score",
            y="recipe_type",
            orientation="h",
            title="Unified Food Performance Score by Recipe",
            text_auto=".1f"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            recipe_summary,
            x="satisfaction",
            y="cancellation_rate",
            size="customers",
            color="recipe_type",
            title="Satisfaction vs Cancellation by Recipe",
            hover_data=["repeat_rate", "food_score"]
        )
        st.plotly_chart(fig, use_container_width=True)


with tabs[1]:
    st.subheader("Food Performance Intelligence")

    recipe_scorecard = (
        filtered.groupby("recipe_type")
        .agg(
            customers=("customer_id", "count"),
            avg_food_performance_score=("food_performance_score", "mean"),
            cancellation_rate=("cancelled", "mean"),
            repeat_rate=("repeat_order", "mean"),
            switch_rate=("flavour_switched", "mean"),
            complaint_rate=("complaint", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            high_nir_rate=("high_nir_anomaly", "mean")
        )
        .reset_index()
        .sort_values("avg_food_performance_score", ascending=False)
    )

    st.dataframe(recipe_scorecard, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            recipe_scorecard,
            x="recipe_type",
            y="cancellation_rate",
            title="Cancellation Rate by Recipe",
            text_auto=".1%"
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            recipe_scorecard,
            x="recipe_type",
            y="avg_satisfaction",
            title="Average Satisfaction by Recipe",
            text_auto=".2f"
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


with tabs[2]:
    st.subheader("Customer Behaviour & Retention Analytics")

    behaviour_summary = (
        filtered.groupby(["flavour_switched", "complaint"])
        .agg(
            customers=("customer_id", "count"),
            cancellation_rate=("cancelled", "mean"),
            repeat_rate=("repeat_order", "mean"),
            avg_satisfaction=("satisfaction_score", "mean")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            behaviour_summary,
            x="flavour_switched",
            y="cancellation_rate",
            color="complaint",
            barmode="group",
            title="Cancellation Rate by Behaviour Signals",
            text_auto=".1%"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered,
            x="time_to_event_days",
            color="cancelled",
            nbins=40,
            title="Time-to-Event Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    segment = (
        filtered.groupby(["dog_size", "sensitive_stomach"])
        .agg(
            customers=("customer_id", "count"),
            cancellation_rate=("cancelled", "mean"),
            switch_rate=("flavour_switched", "mean"),
            repeat_rate=("repeat_order", "mean"),
            avg_satisfaction=("satisfaction_score", "mean")
        )
        .reset_index()
        .sort_values("cancellation_rate", ascending=False)
    )

    st.markdown("### Behavioural Risk Segments")
    st.dataframe(segment, use_container_width=True)


with tabs[3]:
    st.subheader("Experimentation & Recommendation Version Evaluation")

    experiment_summary = (
        filtered.groupby("recommendation_version")
        .agg(
            customers=("customer_id", "count"),
            cancellation_rate=("cancelled", "mean"),
            repeat_rate=("repeat_order", "mean"),
            switch_rate=("flavour_switched", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_food_score=("food_performance_score", "mean")
        )
        .reset_index()
    )

    st.dataframe(experiment_summary, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.bar(experiment_summary, x="recommendation_version", y="cancellation_rate", title="Cancellation by Version", text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(experiment_summary, x="recommendation_version", y="avg_satisfaction", title="Satisfaction by Version", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.bar(experiment_summary, x="recommendation_version", y="avg_food_score", title="Food Score by Version", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)


with tabs[4]:
    st.subheader("QC / NIR Manufacturing Intelligence")

    qc_summary = (
        filtered.assign(
            qc_group=np.where(filtered["nir_anomaly_score"] >= 2.0, "High NIR anomaly", "Normal NIR")
        )
        .groupby("qc_group")
        .agg(
            customers=("customer_id", "count"),
            cancellation_rate=("cancelled", "mean"),
            complaint_rate=("complaint", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_food_score=("food_performance_score", "mean")
        )
        .reset_index()
    )

    st.dataframe(qc_summary, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            filtered,
            x="moisture_deviation",
            y="nir_anomaly_score",
            color="complaint",
            size="satisfaction_score",
            title="NIR Anomaly vs Moisture Deviation"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered,
            x="high_nir_anomaly",
            y="satisfaction_score",
            points="outliers",
            title="Satisfaction by NIR Anomaly"
        )
        st.plotly_chart(fig, use_container_width=True)


with tabs[5]:
    st.subheader("Customer Feedback Intelligence")

    feedback_summary = (
        filtered.groupby(["feedback_sentiment", "feedback_topic"])
        .agg(
            customers=("customer_id", "count"),
            cancellation_rate=("cancelled", "mean"),
            complaint_rate=("complaint", "mean"),
            avg_satisfaction=("satisfaction_score", "mean")
        )
        .reset_index()
        .sort_values("customers", ascending=False)
    )

    st.dataframe(feedback_summary, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.treemap(
            feedback_summary,
            path=["feedback_sentiment", "feedback_topic"],
            values="customers",
            color="cancellation_rate",
            title="Feedback Themes by Sentiment and Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            feedback_summary.head(15),
            x="feedback_topic",
            y="cancellation_rate",
            color="feedback_sentiment",
            title="Cancellation by Feedback Topic",
            text_auto=".1%"
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


with tabs[6]:
    st.subheader("AI Recommendation Optimisation Engine")

    st.markdown(
        "Select a dog profile. The engine scores eligible recipes and recommends the recipe "
        "with the highest predicted food performance score."
    )

    recipes = df[["recipe_type", "protein_pct", "fat_pct", "fibre_pct"]].drop_duplicates().reset_index(drop=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dog_size = st.selectbox("Dog Size", sorted(df["dog_size"].unique()))
    with col2:
        dog_age = st.slider("Dog Age", 0.3, 15.0, 5.0, 0.1)
    with col3:
        activity = st.selectbox("Activity Level", sorted(df["activity_level"].unique()))
    with col4:
        sensitive = st.selectbox("Sensitive Stomach", [0, 1])

    breed = st.selectbox("Breed Group", sorted(df["breed_group"].unique()))

    scenario_rows = []

    for _, recipe in recipes.iterrows():
        scenario_rows.append({
            "dog_size": dog_size,
            "dog_age_years": dog_age,
            "activity_level": activity,
            "sensitive_stomach": sensitive,
            "breed_group": breed,
            "recipe_type": recipe["recipe_type"],
            "protein_pct": recipe["protein_pct"],
            "fat_pct": recipe["fat_pct"],
            "fibre_pct": recipe["fibre_pct"],
            "nir_anomaly_score": df["nir_anomaly_score"].mean(),
            "moisture_deviation": 0,
            "protein_deviation": 0,
            "fat_deviation": 0,
            "qc_pass": 1
        })

    scenario = pd.DataFrame(scenario_rows)
    scenario["predicted_food_performance_score"] = score_model.predict(scenario[score_features])

    scenario["allowed"] = 1
    scenario.loc[(scenario["recipe_type"] == "Puppy Growth") & (scenario["dog_age_years"] >= 2), "allowed"] = 0
    scenario.loc[(scenario["recipe_type"] == "Turkey Senior") & (scenario["dog_age_years"] < 7), "allowed"] = 0

    allowed = scenario[scenario["allowed"] == 1].copy()
    best = allowed.sort_values("predicted_food_performance_score", ascending=False).iloc[0]

    st.success(
        f"Recommended recipe: {best['recipe_type']} | "
        f"Predicted Food Performance Score: {best['predicted_food_performance_score']:.1f}"
    )

    fig = px.bar(
        allowed.sort_values("predicted_food_performance_score"),
        x="predicted_food_performance_score",
        y="recipe_type",
        orientation="h",
        title="Predicted Score by Eligible Recipe",
        text_auto=".1f"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(allowed.sort_values("predicted_food_performance_score", ascending=False), use_container_width=True)


with tabs[7]:
    st.subheader("Scenario Simulator")

    col1, col2, col3 = st.columns(3)

    with col1:
        churn_reduction = st.slider("Relative cancellation reduction", 0.0, 0.30, 0.08, 0.01)
    with col2:
        repeat_increase = st.slider("Relative repeat-rate increase", 0.0, 0.30, 0.05, 0.01)
    with col3:
        satisfaction_gain = st.slider("Satisfaction gain", 0.0, 0.50, 0.12, 0.01)

    current_churn = filtered["cancelled"].mean()
    current_repeat = filtered["repeat_order"].mean()
    current_satisfaction = filtered["satisfaction_score"].mean()

    simulated_churn = current_churn * (1 - churn_reduction)
    simulated_repeat = min(current_repeat * (1 + repeat_increase), 1)
    simulated_satisfaction = current_satisfaction + satisfaction_gain

    sim = pd.DataFrame({
        "Metric": ["Cancellation Rate", "Repeat Order Rate", "Average Satisfaction"],
        "Current": [current_churn, current_repeat, current_satisfaction],
        "Simulated": [simulated_churn, simulated_repeat, simulated_satisfaction]
    })

    sim["Change"] = sim["Simulated"] - sim["Current"]

    st.dataframe(sim, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sim["Metric"], y=sim["Current"], name="Current"))
    fig.add_trace(go.Bar(x=sim["Metric"], y=sim["Simulated"], name="Simulated"))
    fig.update_layout(title="Current vs Simulated Scenario", barmode="group")
    st.plotly_chart(fig, use_container_width=True)


with tabs[8]:
    st.subheader("Machine Learning Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn Prediction Model")
        st.metric("Accuracy", f"{churn_metrics['accuracy']:.3f}")
        st.metric("ROC-AUC", f"{churn_metrics['roc_auc']:.3f}")

    with col2:
        st.markdown("### Food Performance Score Model")
        st.metric("MAE", f"{score_metrics['MAE']:.3f}")
        st.metric("R²", f"{score_metrics['R2']:.3f}")

    st.info(
        "This dashboard uses simulated data to demonstrate the analytical architecture. "
        "With real data, the same platform can support production-grade food performance intelligence."
    )
st.markdown("---")

st.markdown("""
Developed by **Dr Muhammad Zahir Khan**  
Applied Statistics | Product Analytics | Machine Learning | Decision Science  

Tools Used:
Python • SQL • Streamlit • Plotly • SQLite • Scikit-Learn • Survival Analysis • NLP • Experimentation Analytics
""")
