import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Burnout Risk Predictor",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------------
# DEFINE COLUMNS (same as training)
# ---------------------------------------------------------
NUMERIC_COLS = [
    "age", "experience_years", "work_hours_per_week", "overtime_hours",
    "meetings_per_day", "deadlines_missed", "job_satisfaction",
    "manager_support", "work_life_balance", "sleep_hours",
    "physical_activity_days", "screen_time_hours", "caffeine_intake",
    "social_support_score", "stress_level", "anxiety_score", "depression_score"
]
CATEGORICAL_COLS = ["gender", "job_role", "company_size", "work_mode"]

# ---------------------------------------------------------
# LOAD MODEL (no preprocessor pkl needed!)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("burnout_model.pkl")
    le    = joblib.load("label_encoder.pkl")
    # load training data stats for scaling
    stats = joblib.load("feature_stats.pkl")
    cats  = joblib.load("feature_cats.pkl")
    return model, le, stats, cats

try:
    model, label_encoder, feature_stats, feature_cats = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model files not found. Error: {e}")

# ---------------------------------------------------------
# MANUAL PREPROCESSING FUNCTION
# (replicates what the pipeline did during training)
# ---------------------------------------------------------
def preprocess_input(input_df, stats, cats):
    result = []

    # Numeric: fill missing with median, then standardize
    for col in NUMERIC_COLS:
        val = input_df[col].values[0]
        mean = stats[col]["mean"]
        std  = stats[col]["std"]
        scaled = (val - mean) / std if std > 0 else 0.0
        result.append(scaled)

    # Categorical: one-hot encode using training categories
    for col in CATEGORICAL_COLS:
        val = input_df[col].values[0]
        for cat in cats[col]:
            result.append(1.0 if val == cat else 0.0)

    return np.array(result).reshape(1, -1)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🔥 Employee Burnout Risk Predictor")
st.markdown("Fill in the employee details on the left and click **Predict** to assess burnout risk.")
st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
st.sidebar.header("👤 Employee Details")
st.sidebar.markdown("Adjust the values below:")

st.sidebar.subheader("Demographic Info")
age    = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])

st.sidebar.subheader("Job Info")
job_role = st.sidebar.selectbox("Job Role", [
    "Backend Developer", "Frontend Developer", "DevOps",
    "Data Scientist", "Software Engineer",
    "Product Manager", "QA Engineer", "ML Engineer"
])
experience_years = st.sidebar.slider("Experience (Years)", 0.0, 40.0, 3.0, step=0.5)
company_size     = st.sidebar.selectbox("Company Size", ["Startup", "Mid-size", "Large", "MNC"])
work_mode        = st.sidebar.selectbox("Work Mode", ["Remote", "Hybrid", "Onsite"])

st.sidebar.subheader("Workload")
work_hours_per_week = st.sidebar.slider("Work Hours per Week", 20.0, 90.0, 45.0)
overtime_hours      = st.sidebar.slider("Overtime Hours per Week", 0.0, 30.0, 2.0, step=0.5)
meetings_per_day    = st.sidebar.slider("Meetings per Day", 0.0, 15.0, 4.0, step=0.5)
deadlines_missed    = st.sidebar.slider("Deadlines Missed (last month)", 0, 20, 1)

st.sidebar.subheader("Wellbeing")
job_satisfaction       = st.sidebar.slider("Job Satisfaction (1-10)", 1.0, 10.0, 6.0, step=0.1)
manager_support        = st.sidebar.slider("Manager Support (1-10)", 1.0, 10.0, 5.0, step=0.1)
work_life_balance      = st.sidebar.slider("Work-Life Balance (1-10)", 1.0, 10.0, 5.0, step=0.1)
sleep_hours            = st.sidebar.slider("Sleep Hours per Night", 2.0, 10.0, 7.0, step=0.5)
physical_activity_days = st.sidebar.slider("Physical Activity Days/Week", 0, 7, 2)
screen_time_hours      = st.sidebar.slider("Screen Time Hours/Day", 1.0, 18.0, 8.0, step=0.5)
caffeine_intake        = st.sidebar.slider("Caffeine Intake (cups/day)", 0, 10, 2)
social_support_score   = st.sidebar.slider("Social Support Score (1-10)", 1.0, 10.0, 5.0, step=0.1)

st.sidebar.subheader("Mental Health")
stress_level     = st.sidebar.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, step=0.1)
anxiety_score    = st.sidebar.slider("Anxiety Score (1-10)", 1.0, 10.0, 4.0, step=0.1)
depression_score = st.sidebar.slider("Depression Score (1-10)", 1.0, 10.0, 3.0, step=0.1)

# ---------------------------------------------------------
# BUILD INPUT DATAFRAME
# ---------------------------------------------------------
input_data = {
    "age": age, "experience_years": experience_years,
    "work_hours_per_week": work_hours_per_week, "overtime_hours": overtime_hours,
    "meetings_per_day": meetings_per_day, "deadlines_missed": deadlines_missed,
    "job_satisfaction": job_satisfaction, "manager_support": manager_support,
    "work_life_balance": work_life_balance, "sleep_hours": sleep_hours,
    "physical_activity_days": physical_activity_days,
    "screen_time_hours": screen_time_hours, "caffeine_intake": caffeine_intake,
    "social_support_score": social_support_score, "stress_level": stress_level,
    "anxiety_score": anxiety_score, "depression_score": depression_score,
    "gender": gender, "job_role": job_role,
    "company_size": company_size, "work_mode": work_mode,
}
input_df = pd.DataFrame([input_data])

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")

    if model_loaded:
        if st.button("🔍 Predict Burnout Risk", use_container_width=True):

            input_preprocessed = preprocess_input(input_df, feature_stats, feature_cats)

            prediction_encoded = model.predict(input_preprocessed)[0]
            prediction_label   = label_encoder.inverse_transform([prediction_encoded])[0]
            probabilities      = model.predict_proba(input_preprocessed)[0]

            if prediction_label == "High":
                st.error("### 🔴 Risk Level: HIGH")
                st.error(
                    "⚠️ This employee is at HIGH risk of burnout.\n\n"
                    "**Recommended Action:** Schedule an immediate 1-on-1, "
                    "review workload urgently, and connect with mental health resources."
                )
            elif prediction_label == "Moderate":
                st.warning("### 🟠 Risk Level: MODERATE")
                st.warning(
                    "This employee shows moderate burnout risk.\n\n"
                    "**Recommended Action:** Monitor closely, schedule a wellness "
                    "check-in, and consider workload adjustments."
                )
            else:
                st.success("### 🟢 Risk Level: LOW")
                st.success(
                    "This employee currently shows low burnout risk.\n\n"
                    "**Recommended Action:** Continue regular engagement and "
                    "maintain current support structures."
                )

            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                "Risk Level": label_encoder.classes_,
                "Probability (%)": (probabilities * 100).round(2)
            }).sort_values("Probability (%)", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = {"High": "red", "Moderate": "orange", "Low": "green"}
            bar_colors = [colors.get(lvl, "gray") for lvl in prob_df["Risk Level"]]
            ax.barh(prob_df["Risk Level"], prob_df["Probability (%)"], color=bar_colors)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Burnout Risk Probability")
            ax.set_xlim(0, 100)
            for i, val in enumerate(prob_df["Probability (%)"]):
                ax.text(val + 1, i, f"{val:.1f}%", va="center")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Model files not found. Please upload all pkl files to GitHub.")

st.markdown("---")
st.caption(
    "Note: This tool is designed to support HR decisions, not replace human judgment. "
    "Always involve a qualified professional before taking action."
)
