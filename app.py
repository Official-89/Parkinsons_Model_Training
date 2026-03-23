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
        dataset = pd.read_csv('MDVR_all_features.csv')
        return model, sc, dataset
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

svm_model, scaler, df = load_assets()

st.set_page_config(page_title="Parkinson's Case Study", layout="wide")

st.title("🧠 Parkinson's Disease Detection: Case Study")
st.markdown("---")

if df is not None:
    # --- AUTO-FIX LOGIC ---
    # We need to find exactly which columns the scaler expects.
    # Most voice datasets use only numeric columns after dropping IDs.
    
    all_numeric_features = df.select_dtypes(include=[np.number]).drop(columns=['status'], errors='ignore')
    
    # Check how many features the scaler was trained on
    expected_count = scaler.n_features_in_
    actual_count = all_numeric_features.shape[1]

    if actual_count != expected_count:
        st.warning(f"⚠️ **Feature Mismatch:** Your model expects **{expected_count}** features, but your CSV has **{actual_count}** numeric columns.")
        st.write("I will automatically use the first ", expected_count, " numeric features to try and match.")
        # This takes the first N columns to match the scaler
        final_features = all_numeric_features.iloc[:, :expected_count]
    else:
        final_features = all_numeric_features

    # --- SECTION 2: EXPLORER ---
    row_idx = st.slider("Select Patient Record Index", 0, len(df)-1, 0)
    selected_row_full = df.iloc[[row_idx]]
    selected_features = final_features.iloc[[row_idx]]
    
    st.write("### Patient Metrics (Aligned for Model)")
    st.dataframe(selected_features)

    if st.button("Run Diagnostic"):
        # Now the shape will match perfectly
        scaled_input = scaler.transform(selected_features)
        prediction = svm_model.predict(scaled_input)
        
        actual = "Parkinson's" if selected_row_full['status'].values[0] == 1 else "Healthy"
        
        if prediction[0] == 1:
            st.error(f"**AI Result:** Parkinson's Detected | **Actual:** {actual}")
        else:
            st.success(f"**AI Result:** Healthy | **Actual:** {actual}")

    # --- SECTION 3: MANUAL ENTRY ---
    st.divider()
    st.subheader("⌨️ Manual Requirement Entry")
    with st.expander("Modify Metrics Manually"):
        manual_df = selected_features.copy()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            # Using the first numeric column (usually Frequency)
            f0 = st.number_input("Voice Frequency (Hz)", value=float(manual_df.iloc[0, 0]))
        with c2:
            # Using the Jitter column (usually index 3 or 4)
            jitter = st.number_input("Jitter Metrics", value=float(manual_df.iloc[0, 3]), format="%.5f")
        with c3:
            # Using Shimmer (usually index 8)
            shimmer = st.number_input("Shimmer Metrics", value=float(manual_df.iloc[0, 8]), format="%.5f")

        manual_df.iloc[0, 0] = f0
        manual_df.iloc[0, 3] = jitter
        manual_df.iloc[0, 8] = shimmer

        if st.button("Predict Manual Input"):
            m_scaled = scaler.transform(manual_df)
            m_pred = svm_model.predict(m_scaled)
            st.info("Result: " + ("Parkinson's Positive" if m_pred[0] == 1 else "Healthy"))
