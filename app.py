import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the "Engine" (Model, Scaler, and Dataset)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('parkinsons_model.pkl')
        sc = joblib.load('scaler.pkl')
        dataset = pd.read_csv('MDVR_all_features.csv')
        return model, sc, dataset
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

svm_model, scaler, df = load_assets()

# Page Config
st.set_page_config(page_title="Parkinson's Case Study Dashboard", layout="wide")

# --- SECTION 1: RESEARCH OVERVIEW ---
st.title("🧠 Parkinson's Disease Detection: Case Study Dashboard")
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📊 Model Performance Comparison")
    comparison_data = {
        "Algorithm": ["Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"],
        "LOO Accuracy": ["94.87%", "91.28%"],
        "Verdict": ["Optimal Choice", "Baseline Model"]
    }
    st.table(pd.DataFrame(comparison_data))

with col_right:
    st.subheader("📂 Dataset Insights")
    if df is not None:
        st.info(f"**Dataset Source:** MDVR Acoustic Records")
        st.write(f"**Total Records:** {len(df)} samples")
        st.write(f"**Acoustic Features:** {len(df.columns) - 3} (excluding IDs and Status)")

st.markdown("---")

# --- SECTION 2: CASE STUDY EXPLORER ---
if df is not None:
    st.subheader("🔎 Case Study Record Explorer")
    st.write("Select a patient record from the dataset to analyze.")

    # Slider to select a row index
    row_idx = st.slider("Select Patient Record Index", 0, len(df)-1, 0)
    selected_row = df.iloc[[row_idx]]
    
    # CRITICAL: Drop 'voiceID' along with name and status so only numbers remain
    features_only = selected_row.drop(columns=['name', 'status', 'voiceID', 'Unnamed: 0'], errors='ignore')
    
    st.write("Current Patient Metrics:")
    st.dataframe(features_only)

    if st.button("Run Diagnostic on Selected Case"):
        # Scale and Predict
        scaled_input = scaler.transform(features_only)
        prediction = svm_model.predict(scaled_input)
        
        actual_val = selected_row['status'].values[0]
        actual_label = "Parkinson's Positive" if actual_val == 1 else "Healthy / Negative"
        pred_label = "Parkinson's Positive" if prediction[0] == 1 else "Healthy / Negative"
        
        if prediction[0] == 1:
            st.error(f"**AI Result:** {pred_label}  \n**Actual Dataset Label:** {actual_label}")
        else:
            st.success(f"**AI Result:** {pred_label}  \n**Actual Dataset Label:** {actual_label}")

    st.markdown("---")

    # --- SECTION 3: MANUAL REQUIREMENT TESTING ---
    st.subheader("⌨️ Manual Requirement Entry")
    st.write("Modify specific requirements to see how vocal frequency and jitter affect the diagnosis.")

    with st.expander("Modify Metrics Manually"):
        manual_df = features_only.copy()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            # Column 0 is now 'meanF0Hz' because 'voiceID' was dropped
            val_f0 = st.number_input("Fundamental Frequency (Hz)", value=float(manual_df.iloc[0, 0]))
        with c2:
            # Column 3 is 'localJitter'
            val_jitter = st.number_input("Local Jitter", value=float(manual_df.iloc[0, 3]), format="%.5f")
        with c3:
            # Column 7 is 'localShimmer'
            val_shimmer = st.number_input("Local Shimmer", value=float(manual_df.iloc[0, 7]), format="%.5f")

        # Inject manual values back into the feature set
        manual_df.iloc[0, 0] = val_f0
        manual_df.iloc[0, 3] = val_jitter
        manual_df.iloc[0, 7] = val_shimmer

        if st.button("Predict with Manual Requirements"):
            manual_scaled = scaler.transform(manual_df)
            manual_pred = svm_model.predict(manual_scaled)
            
            res = "Parkinson's Positive" if manual_pred[0] == 1 else "Negative / Healthy"
            st.info(f"AI Manual Diagnostic Result: **{res}**")

else:
    st.warning("Ensure 'MDVR_all_features.csv' is in your GitHub repository.")
