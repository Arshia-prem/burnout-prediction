import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Burnout Risk Predictor",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model        = joblib.load("burnout_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    le           = joblib.load("label_encoder.pkl")
    return model, preprocessor, le

try:
    model, preprocessor, label_encoder = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model files not found. Please run the notebook first to generate model/ folder.\nError: {e}")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🔥 Employee Burnout Risk Predictor")
st.markdown("Fill in the employee details on the left and click **Predict** to assess burnout risk.")
st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR — INPUT FORM
# ---------------------------------------------------------
st.sidebar.header("👤 Employee Details")
st.sidebar.markdown("Adjust the values below:")

# --- Demographic Info ---
st.sidebar.subheader("Demographic Info")
age = st.sidebar.slider("Age", min_value=18, max_value=65, value=30, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])

# --- Job Info ---
st.sidebar.subheader("Job Info")
job_role = st.sidebar.selectbox("Job Role", [
    "Backend Developer", "Frontend Developer", "DevOps",
    "Data Scientist", "Software Engineer",
    "Product Manager", "QA Engineer", "ML Engineer"
])
experience_years = st.sidebar.slider("Experience (Years)", 0.0, 40.0, 3.0, step=0.5)
company_size = st.sidebar.selectbox("Company Size", ["Startup", "Mid-size", "Large", "MNC"])
work_mode = st.sidebar.selectbox("Work Mode", ["Remote", "Hybrid", "Onsite"])

# --- Workload ---
st.sidebar.subheader("Workload")
work_hours_per_week = st.sidebar.slider("Work Hours per Week", 20.0, 90.0, 45.0, step=1.0)
overtime_hours      = st.sidebar.slider("Overtime Hours per Week", 0.0, 30.0, 2.0, step=0.5)
meetings_per_day    = st.sidebar.slider("Meetings per Day", 0.0, 15.0, 4.0, step=0.5)
deadlines_missed    = st.sidebar.slider("Deadlines Missed (last month)", 0, 20, 1, step=1)

# --- Wellbeing ---
st.sidebar.subheader("Wellbeing")
job_satisfaction      = st.sidebar.slider("Job Satisfaction (1-10)", 1.0, 10.0, 6.0, step=0.1)
manager_support       = st.sidebar.slider("Manager Support (1-10)", 1.0, 10.0, 5.0, step=0.1)
work_life_balance     = st.sidebar.slider("Work-Life Balance (1-10)", 1.0, 10.0, 5.0, step=0.1)
sleep_hours           = st.sidebar.slider("Sleep Hours per Night", 2.0, 10.0, 7.0, step=0.5)
physical_activity_days= st.sidebar.slider("Physical Activity Days/Week", 0, 7, 2, step=1)
screen_time_hours     = st.sidebar.slider("Screen Time Hours/Day", 1.0, 18.0, 8.0, step=0.5)
caffeine_intake       = st.sidebar.slider("Caffeine Intake (cups/day)", 0, 10, 2, step=1)
social_support_score  = st.sidebar.slider("Social Support Score (1-10)", 1.0, 10.0, 5.0, step=0.1)
has_therapy           = st.sidebar.selectbox("Currently in Therapy?", ["No", "Yes"])

# --- Mental Health Scores ---
st.sidebar.subheader("Mental Health")
stress_level     = st.sidebar.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, step=0.1)
anxiety_score    = st.sidebar.slider("Anxiety Score (1-10)", 1.0, 10.0, 4.0, step=0.1)
depression_score = st.sidebar.slider("Depression Score (1-10)", 1.0, 10.0, 3.0, step=0.1)

# ---------------------------------------------------------
# BUILD INPUT DATAFRAME
# ---------------------------------------------------------
input_data = {
    "age":                    age,
    "experience_years":       experience_years,
    "work_hours_per_week":    work_hours_per_week,
    "overtime_hours":         overtime_hours,
    "meetings_per_day":       meetings_per_day,
    "deadlines_missed":       deadlines_missed,
    "job_satisfaction":       job_satisfaction,
    "manager_support":        manager_support,
    "work_life_balance":      work_life_balance,
    "sleep_hours":            sleep_hours,
    "physical_activity_days": physical_activity_days,
    "screen_time_hours":      screen_time_hours,
    "caffeine_intake":        caffeine_intake,
    "social_support_score":   social_support_score,
    "stress_level":           stress_level,
    "anxiety_score":          anxiety_score,
    "depression_score":       depression_score,
    "gender":                 gender,
    "job_role":               job_role,
    "company_size":           company_size,
    "work_mode":              work_mode,
}

input_df = pd.DataFrame([input_data])

# ---------------------------------------------------------
# MAIN AREA — SHOW INPUT SUMMARY + PREDICTION
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")

    if model_loaded:
        if st.button("🔍 Predict Burnout Risk", use_container_width=True):

            # preprocess input
            input_preprocessed = preprocessor.transform(input_df)

            # predict
            prediction_encoded = model.predict(input_preprocessed)[0]
            prediction_label   = label_encoder.inverse_transform([prediction_encoded])[0]
            probabilities      = model.predict_proba(input_preprocessed)[0]

            # display risk level with color
            if prediction_label == "High":
                st.error(f"### 🔴 Risk Level: HIGH")
                st.error(
                    "⚠️ This employee is at HIGH risk of burnout.\n\n"
                    "**Recommended Action:** Schedule an immediate 1-on-1 with the employee, "
                    "review workload urgently, and connect them with mental health resources."
                )
            elif prediction_label == "Moderate":
                st.warning(f"### 🟠 Risk Level: MODERATE")
                st.warning(
                    "This employee shows moderate burnout risk.\n\n"
                    "**Recommended Action:** Monitor closely, schedule a wellness check-in, "
                    "and consider workload adjustments."
                )
            else:
                st.success(f"### 🟢 Risk Level: LOW")
                st.success(
                    "This employee currently shows low burnout risk.\n\n"
                    "**Recommended Action:** Continue regular engagement. "
                    "Maintain current support structures."
                )

            # probability breakdown chart
            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                "Risk Level": label_encoder.classes_,
                "Probability (%)": (probabilities * 100).round(2)
            }).sort_values("Probability (%)", ascending=False)

            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = {"High": "red", "Moderate": "orange", "Low": "green"}
            bar_colors = [colors[lvl] for lvl in prob_df["Risk Level"]]
            ax.barh(prob_df["Risk Level"], prob_df["Probability (%)"], color=bar_colors)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Burnout Risk Probability")
            ax.set_xlim(0, 100)
            for i, (val, label) in enumerate(zip(prob_df["Probability (%)"], prob_df["Risk Level"])):
                ax.text(val + 1, i, f"{val:.1f}%", va="center")
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("Run the Jupyter notebook first to generate the model files, then restart this app.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "**Note:** This tool is designed to **support** HR decisions, not replace human judgment. "
    "Always involve a qualified professional before taking action based on these predictions."
)
