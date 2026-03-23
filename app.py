import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Load Assets (Model, Scaler, and Dataset)
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

# --- PAGE SETUP ---
st.set_page_config(page_title="Parkinson's AI Case Study", layout="wide", page_icon="🧠")

st.title("🧠 Parkinson's Disease Detection: AI Case Study Dashboard")
st.markdown("---")

if df is not None:
    # --- MODEL PERFORMANCE HEADER ---
    st.subheader("📊 Research Phase: Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SVM Accuracy (RBF)", "94.87%", "Winner")
    m2.metric("KNN Accuracy (k=9)", "91.28%", "-3.59%")
    m3.metric("Dataset Size", f"{len(df)} Records")
    m4.metric("Features Analyzed", f"{scaler.n_features_in_} Metrics")

    st.markdown("---")

    # --- SECTION 2: CASE STUDY EXPLORER ---
    st.subheader("🔎 Clinical Record Explorer")
    st.write("Use the slider below to browse through the MDVR patient database.")

    row_idx = st.slider("Select Patient Record (Index)", 0, len(df)-1, 0)
    selected_row_full = df.iloc[[row_idx]]

    # Identify non-feature columns
    target_col = 'label' if 'label' in df.columns else 'status'
    id_col = 'voiceID'
    
    # Isolate numeric features for the model
    features_only = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    features_only = features_only.drop(columns=['Unnamed: 0'], errors='ignore') # Common artifact
    
    # Ensure columns match training count
    expected_count = scaler.n_features_in_
    selected_features = features_only.iloc[[row_idx], :expected_count]

    st.write(f"**Current Patient ID:** `{selected_row_full[id_col].values[0]}`")
    st.dataframe(selected_features, use_container_width=True)

    # --- PREDICTION LOGIC ---
    if st.button("🚀 Run AI Diagnostic", type="primary"):
        scaled_input = scaler.transform(selected_features)
        prediction = svm_model.predict(scaled_input)
        
        # UI Feedback
        actual_val = selected_row_full[target_col].values[0]
        actual_label = "Parkinson's Positive" if actual_val == 1 else "Healthy / Control"
        pred_label = "Parkinson's Positive" if prediction[0] == 1 else "Healthy / Control"

        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("Diagnosis Result")
            if prediction[0] == 1:
                st.error(f"**AI PREDICTION:** {pred_label}")
            else:
                st.success(f"**AI PREDICTION:** {pred_label}")
            
            # Explain mismatches (False Positives/Negatives)
            if pred_label == actual_label:
                st.info("✅ **Validation:** AI Prediction matches the Clinical Label.")
            else:
                st.warning("⚠️ **Mismatch:** This represents a False Positive/Negative. Likely due to outlier vocal features.")

        with res_col2:
            st.subheader("Ground Truth (Actual)")
            st.write(f"This record is clinically labeled as: **{actual_label}**")
            
        # --- FEATURE COMPARISON CHART ---
        st.subheader("📊 Metric Comparison")
        st.write("How this patient compares to the dataset average for key indicators:")
        
        metrics_to_compare = ['meanF0Hz', 'localJitter', 'localShimmer']
        avg_vals = df[metrics_to_compare].mean()
        pat_vals = selected_features[metrics_to_compare].iloc[0]

        comparison_df = pd.DataFrame({
            "Metric": metrics_to_compare * 2,
            "Value": list(avg_vals) + list(pat_vals),
            "Type": ["Dataset Avg"] * 3 + ["This Patient"] * 3
        })
        
        fig = px.bar(comparison_df, x="Metric", y="Value", color="Type", barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- SECTION 3: MANUAL INPUT ---
    st.subheader("⌨️ Manual Requirement Testing")
    with st.expander("Modify Metrics for Sensitivity Testing"):
        manual_df = selected_features.copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            val_f0 = st.number_input("Fundamental Frequency (meanF0Hz)", value=float(manual_df.iloc[0, 0]))
        with c2:
            val_jitter = st.number_input("Local Jitter (%)", value=float(manual_df.iloc[0, 3]), format="%.6f")
        with c3:
            val_shimmer = st.number_input("Local Shimmer", value=float(manual_df.iloc[0, 7]), format="%.6f")

        manual_df.iloc[0, 0] = val_f0
        manual_df.iloc[0, 3] = val_jitter
        manual_df.iloc[0, 7] = val_shimmer

        if st.button("Predict Manual Entry"):
            m_scaled = scaler.transform(manual_df)
            m_pred = svm_model.predict(m_scaled)
            res = "Parkinson's Positive" if m_pred[0] == 1 else "Healthy / Control"
            st.info(f"Manual Diagnostic Result: **{res}**")

else:
    st.warning("Please ensure 'MDVR_all_features.csv' is in your GitHub repository.")
