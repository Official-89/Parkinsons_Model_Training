import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Assets (Model, Scaler, and the Data)
@st.cache_resource
def load_assets():
    model_svm = joblib.load('parkinsons_model.pkl')
    # If you saved a KNN model too, load it here:
    # model_knn = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('parkinsons_data.csv') # Ensure name matches your file
    return model_svm, scaler, df

svm_model, sc, df = load_assets()

st.set_page_config(page_title="Parkinson's Case Study", layout="wide")

# --- SECTION 1: CASE STUDY OVERVIEW ---
st.title("📊 Project Case Study: Parkinson's Voice Analysis")
st.write("This dashboard showcases the detection of Parkinson's Disease using acoustic voice features.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Performance")
    # You can hardcode these based on your LOOCV results from Colab
    results = {
        "Algorithm": ["SVM (RBF Kernel)", "KNN (k=9)"],
        "LOO Accuracy": ["94.8% ", "91.2%"] # Update with your real numbers
    }
    st.table(pd.DataFrame(results))

with col2:
    st.subheader("Dataset Insight")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Features Analyzed: {len(df.columns) - 2}") # Subtracting ID and Label

st.divider()

# --- SECTION 2: SELECT A CASE TO DIAGNOSE ---
st.subheader("🔎 Case Study Explorer")
st.write("Select a patient record from the dataset to see the AI prediction.")

row_idx = st.number_input("Select Row Index (0 to {})".format(len(df)-1), min_value=0, max_value=len(df)-1, value=0)
selected_row = df.iloc[[row_idx]]

# Drop non-feature columns (Adjust these names to match your CSV)
features_only = selected_row.drop(columns=['name', 'status']) 

st.write("Current Patient Metrics:", features_only)

if st.button("Run Model Prediction"):
    scaled_input = sc.transform(features_only)
    prediction = svm_model.predict(scaled_input)
    
    actual_label = "Parkinson's" if selected_row['status'].values[0] == 1 else "Healthy"
    pred_label = "Parkinson's" if prediction[0] == 1 else "Healthy"
    
    if prediction[0] == 1:
        st.error(f"Prediction: {pred_label} (Actual: {actual_label})")
    else:
        st.success(f"Prediction: {pred_label} (Actual: {actual_label})")

st.divider()

# --- SECTION 3: INTERACTIVE REQUIREMENTS (MANUAL INPUT) ---
st.subheader("⌨️ Manual Requirement Testing")
st.write("Adjust features manually to see how the model reacts to different vocal patterns.")

with st.expander("Click to adjust voice features"):
    # Create sliders for the most important features found in your EDA
    f0 = st.slider("MDVP:Fo(Hz)", 70.0, 260.0, 150.0)
    jitter = st.slider("MDVP:Jitter(%)", 0.0, 0.05, 0.01)
    shimmer = st.slider("MDVP:Shimmer", 0.0, 0.15, 0.05)
    
    # Note: For a real prediction, you'd need all features. 
    # For the demo, we can use the selected_row values and only update these 3.
    manual_input = features_only.copy()
    manual_input.iloc[0, 0] = f0 # Update based on your column order
    # ... update others ...

    if st.button("Predict Manual Input"):
        manual_scaled = sc.transform(manual_input)
        manual_pred = svm_model.predict(manual_scaled)
        st.info("Manual Prediction: " + ("Parkinson's" if manual_pred[0]==1 else "Healthy"))
