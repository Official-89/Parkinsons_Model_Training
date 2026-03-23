import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Assets (Cached for speed)
@st.cache_resource
def load_assets():
    svm_model = joblib.load('parkinsons_model.pkl')
    # If you saved a KNN model in Colab, load it here too
    # knn_model = joblib.load('knn_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('MDVR_all_features.csv') 
    return svm_model, scaler, df

svm_model, sc, df = load_assets()

st.set_page_config(page_title="Parkinson's Case Study", layout="wide")

# --- SECTION 1: RESEARCH COMPARISON ---
st.title("📊 Case Study: Parkinson's Disease Detection")
st.write("A comparative analysis of Machine Learning algorithms on the MDVR voice dataset.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Accuracy Comparison")
    # Replace these with the actual % results from your LOOCV in Colab
    comparison_data = {
        "Algorithm": ["SVM (RBF Kernel)", "KNN (Tuned)"],
        "LOO Accuracy (%)": [94.85, 91.20], 
        "Stability (Std Dev)": ["± 2.1%", "± 4.5%"]
    }
    st.table(pd.DataFrame(comparison_data))

with col2:
    st.subheader("Dataset Summary")
    st.info(f"**Total Samples:** {len(df)} | **Total Features:** {len(df.columns) - 2}")
    st.write("Features include Jitter, Shimmer, HNR, and various Frequency spreads.")

st.divider()

# --- SECTION 2: THE CASE EXPLORER ---
st.subheader("🔎 Dataset Row Explorer")
st.write("Select a specific patient record from the dataset to test the model.")

# Slider to pick a row
row_idx = st.slider("Select Record Index", 0, len(df)-1, 10)
selected_row = df.iloc[[row_idx]]

# Show the selected data (dropping non-feature columns for display)
display_data = selected_row.drop(columns=['name', 'status'], errors='ignore')
st.dataframe(display_data)

if st.button("Predict Selected Case"):
    # Ensure we drop 'name' and 'status' before scaling
    features_only = selected_row.drop(columns=['name', 'status'], errors='ignore')
    scaled_input = sc.transform(features_only)
    
    prediction = svm_model.predict(scaled_input)
    actual = "Parkinson's" if selected_row['status'].values[0] == 1 else "Healthy"
    
    if prediction[0] == 1:
        st.error(f"**AI Prediction:** Parkinson's Detected | **Actual Label:** {actual}")
    else:
        st.success(f"**AI Prediction:** Healthy | **Actual Label:** {actual}")

st.divider()

# --- SECTION 3: MANUAL REQUIREMENTS ---
st.subheader("⌨️ Manual Requirement Entry")
st.write("Input specific voice metrics to see how the model reacts.")

with st.expander("Adjust Requirements Manually"):
    # We use the selected row as a base template
    manual_df = features_only.copy()
    
    # Let user adjust 3 key features as a demo
    c1, c2, c3 = st.columns(3)
    with c1:
        new_f0 = st.number_input("Avg Frequency (Hz)", value=float(manual_df.iloc[0, 0]))
    with c2:
        new_jitter = st.number_input("Jitter (%)", value=float(manual_df.iloc[0, 3]))
    with c3:
        new_shimmer = st.number_input("Shimmer", value=float(manual_df.iloc[0, 8]))

    # Update the temp dataframe
    manual_df.iloc[0, 0] = new_f0
    manual_df.iloc[0, 3] = new_jitter
    manual_df.iloc[0, 8] = new_shimmer

    if st.button("Predict Manual Entry"):
        m_scaled = sc.transform(manual_df)
        m_pred = svm_model.predict(m_scaled)
        st.info("Result: " + ("Parkinson's" if m_pred[0] == 1 else "Healthy"))
