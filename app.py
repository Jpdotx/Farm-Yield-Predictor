import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px # Import plotly.express for visualization

# --- 1. SETTINGS & MODEL ---
st.set_page_config(page_title="Farm Analytics Suite", layout="wide", page_icon="🚜")
st.title("🚜 Farm Resilience & ROI Intelligence System")
st.markdown("---")

# Load the brain of the app
try:
    model = joblib.load('farm_yield_model.pkl')
except:
    st.error("Error: 'farm_yield_model.pkl' not found. Please run the training cell in Colab first.")

# --- 2. SIDEBAR: THE BUSINESS LAYER ---
st.sidebar.header("📊 Financial ROI Parameters")
st.sidebar.info("Adjust these to calculate real-world profitability.")
price_per_unit = st.sidebar.number_input("Market Price (per Yield unit)", value=150.0)
cost_per_acre = st.sidebar.number_input("Operational Cost (per Acre)", value=800.0)

# --- 3. MAIN INTERFACE: THE COMPARISON ENGINE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Current Plot Conditions")
    f1 = st.slider("Fertilizer Usage", 0, 100, 50, key="f1")
    t1 = st.slider("Ambient Temperature (°C)", 10, 50, 25, key="t1")
    n1 = st.slider("Nitrogen (N)", 0, 100, 30, key="n1")
    p1 = st.slider("Phosphorus (P)", 0, 100, 30, key="p1")
    k1 = st.slider("Potassium (K)", 0, 100, 30, key="k1")

    # Calculation 1
    yield1 = model.predict([[f1, t1, n1, p1, k1]])[0]
    profit1 = (yield1 * price_per_unit) - cost_per_acre

    st.metric("Predicted Yield", f"{yield1:.2f} Tons")
    st.metric("Net Profit", f"${profit1:.2f}")

with col2:
    st.subheader("💡 Proposed Strategy")
    f2 = st.slider("Fertilizer Usage", 0, 100, 50, key="f2")
    t2 = st.slider("Ambient Temperature (°C)", 10, 50, 25, key="t2")
    n2 = st.slider("Nitrogen (N)", 0, 100, 45, key="n2")
    p2 = st.slider("Phosphorus (P)", 0, 100, 30, key="p2")
    k2 = st.slider("Potassium (K)", 0, 100, 50, key="k2")

    # Calculation 2
    yield2 = model.predict([[f2, t2, n2, p2, k2]])[0]
    profit2 = (yield2 * price_per_unit) - cost_per_acre

    # Deltas (Difference)
    y_delta = yield2 - yield1
    p_delta = profit2 - profit1

    st.metric("Predicted Yield", f"{yield2:.2f} Tons", delta=f"{y_delta:.2f}")
    st.metric("Net Profit", f"${profit2:.2f}", delta=f"${p_delta:.2f}")

# --- 4. CLIMATE STRESS MONITOR ---
st.markdown("---")
st.subheader("🚨 Real-Time Risk Assessment")
avg_temp = (t1 + t2) / 2

if avg_temp > 33:
    st.error(f"🔥 CRITICAL HEAT STRESS: Current temp is {avg_temp}°C. Yield is dropping significantly. Immediate irrigation and shade-netting required.")
elif avg_temp > 28:
    st.warning(f"⚠️ MODERATE CLIMATE RISK: Temp is {avg_temp}°C. Productivity is peaking but evaporation levels are high.")
else:
    st.success(f"✅ OPTIMAL CONDITIONS: Temp is {avg_temp}°C. The farm is in the 'Goldilocks' zone for growth.")

# --- 5. THE CONSULTANT'S PRESCRIPTION ---
st.markdown("---")
st.header("📋 AI-Driven Consultant's Prescription")

if yield2 > yield1:
    st.info(f"✅ STRATEGY VALIDATED: Your proposal increases total harvest by {y_delta:.2f} units.")

    if n2 > n1 and k2 > k1:
        st.write("👉 **Nutrient Sync:** The simultaneous increase in N and K is creating a synergistic effect on the plant's vascular strength.")

    if t2 > t1:
        st.write("👉 **Climate Buffer:** You are pushing for higher yield in a hotter environment. I recommend a 15% increase in watering frequency to support this growth.")
else:
    st.write("Adjust the **Proposed Strategy** sliders to see how different nutrient combinations impact your ROI.")

# --- 6. FEATURE IMPORTANCE VISUALIZATION (Re-added) ---
st.markdown("---")
st.subheader("💡 Strategic Drivers: What influences harvest most?")
try:
    # Assuming 'model' is your trained RandomForestRegressor
    importances = model.feature_importances_
    features = ['Fertilizer', 'temp', 'N', 'P', 'K'] # Ensure this matches your model's feature order
    feat_importances = pd.Series(importances, index=features).sort_values()
    fig = px.bar(feat_importances, orientation='h', color=feat_importances, color_continuous_scale='Greens', title='Feature Importance for Yield Prediction')
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not display feature importance: {e}")
    # 4. CSV Batch Processing Logic
if uploaded_file is not None:
    st.markdown("---")
    st.header("📊 Batch Analysis Results")
    df = pd.read_csv(uploaded_file)
   
    # Required columns check
    req = ['Fertilizer', 'temp', 'N', 'P', 'K']
    if all(col in df.columns for col in req):
        df['Predicted_Yield'] = model.predict(df[req])
        df['Net_Profit'] = (df['Predicted_Yield'] * price) - cost
        st.dataframe(df)
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Prediction Report", data=csv, file_name="farm_report.csv")
    else:
        st.error(f"CSV must have these columns: {req}")

# --- 7. EXPORT & NEXT STEPS ---
st.sidebar.markdown("---")

# Initialize session state for report content if not present
if 'report_content' not in st.session_state:
    st.session_state.report_content = None

generate_button_clicked = st.sidebar.button("Generate Strategy Report")

if generate_button_clicked:
    report_content = f"Farm Strategy Report\n\n" \
                     f"Current Conditions:\n" \
                     f"  Fertilizer: {f1}, Temperature: {t1}°C, N: {n1}, P: {p1}, K: {k1}\n" \
                     f"  Predicted Yield: {yield1:.2f} Tons, Net Profit: ${profit1:.2f}\n\n" \
                     f"Proposed Strategy:\n" \
                     f"  Fertilizer: {f2}, Temperature: {t2}°C, N: {n2}, P: {p2}, K: {k2}\n" \
                     f"  Predicted Yield: {yield2:.2f} Tons, Net Profit: ${profit2:.2f}\n\n" \
                     f"Yield Change: {y_delta:.2f} Tons\n" \
                     f"Profit Change: ${p_delta:.2f}\n\n" \
                     f"Climate Assessment: "
    if avg_temp > 33:
        report_content += "CRITICAL HEAT STRESS\n"
    elif avg_temp > 28:
        report_content += "MODERATE CLIMATE RISK\n"
    else:
        report_content += "OPTIMAL CONDITIONS\n"

    if yield2 > yield1:
        report_content += f"\nStrategy Recommendation: Your proposal increases total harvest by {y_delta:.2f} units.\n"
        if n2 > n1 and k2 > k1:
            report_content += "  - Nutrient Sync: The simultaneous increase in N and K is creating a synergistic effect.\n"
        if t2 > t1:
            report_content += "  - Climate Buffer: Pushing for higher yield in hotter environment, recommend 15% increase in watering frequency.\n"
    else:
        report_content += "\nNo significant improvement with proposed strategy. Adjust sliders for better ROI.\n"

    st.session_state.report_content = report_content
    st.sidebar.success("Strategy Report Generated! Click 'Download Strategy Report' below.")

# Display download button if report content exists in session state
if st.session_state.report_content:
    st.sidebar.download_button(
        label="Download Strategy Report",
        data=st.session_state.report_content,
        file_name="Farm_Strategy_Report.txt",
        mime="text/plain"
    )
