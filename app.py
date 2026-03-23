import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('parkinsons_model.pkl')
        sc = joblib.load('scaler.pkl')
        # Load your specific CSV
        dataset = pd.read_csv('MDVR_all_features.csv')
        return model, sc, dataset
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

svm_model, scaler, df = load_assets()

st.set_page_config(page_title="Parkinson's Case Study Dashboard", layout="wide")

st.title("🧠 Parkinson's Disease Detection: Case Study")
st.markdown("---")

if df is not None:
    # --- DYNAMIC TARGET COLUMN FINDER ---
    # Based on your file, the target is named 'label'
    target_col = None
    for col in ['label', 'status', 'Status', 'class']:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.error("❌ Could not find a 'label' or 'status' column in your CSV.")
        st.stop()

    # --- FEATURE ALIGNMENT ---
    # 1. Get only numeric columns
    # 2. Drop the target column ('label')
    # 3. Drop any index columns like 'Unnamed: 0'
    features_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    features_df = features_df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Ensure the number of features matches what the scaler expects
    expected_count = scaler.n_features_in_
    final_features_df = features_df.iloc[:, :expected_count]

    # --- SECTION 2: CASE EXPLORER ---
    st.subheader("🔎 Case Study Record Explorer")
    row_idx = st.slider("Select Patient Record Index", 0, len(df)-1, 0)
    
    selected_row_full = df.iloc[[row_idx]]
    selected_features = final_features_df.iloc[[row_idx]]
    
    # Show patient info
    st.write(f"**Patient ID:** {selected_row_full['voiceID'].values[0]}")
    st.dataframe(selected_features)

    if st.button("Run Diagnostic on Selected Case"):
        scaled_input = scaler.transform(selected_features)
        prediction = svm_model.predict(scaled_input)
        
        actual_val = selected_row_full[target_col].values[0]
        actual_label = "Parkinson's Positive" if actual_val == 1 else "Healthy"
        pred_label = "Parkinson's Positive" if prediction[0] == 1 else "Healthy"
        
        if prediction[0] == 1:
            st.error(f"**AI Prediction:** {pred_label} | **Actual Record:** {actual_label}")
        else:
            st.success(f"**AI Prediction:** {pred_label} | **Actual Record:** {actual_label}")

    # --- SECTION 3: MANUAL ENTRY ---
    st.divider()
    st.subheader("⌨️ Manual Metric Testing")
    with st.expander("Modify Metrics Manually"):
        manual_df = selected_features.copy()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            # Column 0 is meanF0Hz
            f0 = st.number_input("Avg Frequency (Hz)", value=float(manual_df.iloc[0, 0]))
        with c2:
            # Column 3 is localJitter
            jitter = st.number_input("Local Jitter", value=float(manual_df.iloc[0, 3]), format="%.5f")
        with c3:
            # Column 7 is localShimmer
            shimmer = st.number_input("Local Shimmer", value=float(manual_df.iloc[0, 7]), format="%.5f")

        manual_df.iloc[0, 0] = f0
        manual_df.iloc[0, 3] = jitter
        manual_df.iloc[0, 7] = shimmer

        if st.button("Predict Manual Input"):
            m_scaled = scaler.transform(manual_df)
            m_pred = svm_model.predict(m_scaled)
            res = "Parkinson's Positive" if m_pred[0] == 1 else "Healthy"
            st.info(f"Manual Diagnostic: **{res}**")

else:
    st.warning("Ensure 'MDVR_all_features.csv' is uploaded to your GitHub repository.")
