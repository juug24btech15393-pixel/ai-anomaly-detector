import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import json
from datetime import datetime
from google import genai  # <-- Switched to Google's official SDK

# Set page title and layout
st.set_page_config(page_title="AI Industrial Anomaly Detector", layout="wide")

st.title("🏭 AI Industrial Anomaly Detector with Google Gemma & OCSF")
st.write("This dashboard normalizes AI anomaly alerts into the official **OCSF Security Schema**, then passes the data to **Google Gemma AI** to build plain-text incident mitigation logs.")

# ----------------- LLM EXPLANATION FUNCTION -----------------
def generate_gemma_report(ocsf_log_json):
    """Sends the OCSF log payload to Google Gemma to write an operations summary."""
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ **Gemma Engine Paused**: Please add your GEMINI_API_KEY to the Streamlit Secret Manager to enable live reporting."
        
    try:
        # Initialize Google GenAI client using secrets management
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        
        prompt = f"""
        You are an expert Industrial Cybersecurity Analyst. 
        Review this standardized OCSF (Open Cybersecurity Schema Framework) Detection Finding log from a factory sensor:
        {ocsf_log_json}
        
        Write a brief, professional incident mitigation overview for factory operators.
        Your response must include:
        1. **Summary**: A simple explanation of what happened based on the resource metrics.
        2. **Risk Assessment**: How this specific anomaly impacts production safety.
        3. **Mitigation Blueprint**: 2 steps the local engineering team should take immediately.
        Keep it direct, readable for non-coders, and concise.
        """
        
        # Calling Google's native gemma-2-9b-it model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Could not connect to Gemma engine: {str(e)}"

# ----------------- DATA GENERATION & MODEL TRAINING -----------------
@st.cache_resource
def train_model():
    timestamps = pd.date_range(start="2026-06-11 00:00", periods=1000, freq="min")
    np.random.seed(27)
    temperature = 25 + np.random.normal(0, 1.5, size=1000)
    
    # Inject historical spikes
    temperature[150:155] = 82.5
    temperature[500:506] = 79.1
    temperature[850:853] = 85.0

    df_history = pd.DataFrame({"Timestamp": timestamps, "Temperature_Celsius": temperature})
    
    model = IsolationForest(contamination=0.015, random_state=42)
    model.fit(df_history[['Temperature_Celsius']])
    df_history['AI_Guess'] = model.predict(df_history[['Temperature_Celsius']])
    
    return model, df_history

ai_brain, df = train_model()

# ----------------- SECTION 1: HISTORICAL DATA VIEW -----------------
st.header("Historical Sensor Analysis")
anomalies = df[df['AI_Guess'] == -1]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['Timestamp'], df['Temperature_Celsius'], color='black', label='Normal Reading', alpha=0.4)
ax.scatter(anomalies['Timestamp'], anomalies['Temperature_Celsius'], color='red', label='AI Flagged Spikes', zorder=5)
ax.set_title("Historical AI Training Logs")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ----------------- SECTION 2: LIVE SIMULATION -----------------
st.markdown("---")
st.header("⚡ Live Testing & OCSF Log Generator")
st.write("Drag the slider to alter physical device temperature values.")

test_temp = st.slider("Current Machine Sensor Value (°C):", min_value=15.0, max_value=100.0, value=25.0, step=0.1)

guess = ai_brain.predict([[test_temp]])[0]

if guess == -1:
    st.error(f"⚠️ SIEM ALERT TRIGGERED! Abnormal Thermal Metric: {test_temp}°C")
    
    # Standardized Open Cybersecurity Schema Framework Object
    ocsf_log = {
        "metadata": {
            "version": "1.3.0",
            "product": {"vendor": "FactoryAI", "name": "Thermal Anomaly Engine", "version": "1.0.0"}
        },
        "time": int(datetime.now().timestamp()),
        "class_uid": 2004,
        "class_name": "Detection Finding",
        "category_uid": 2,
        "category_name": "Findings",
        "activity_id": 1, 
        "severity_id": 5, 
        "severity": "Critical",
        "finding_info": {
            "title": "AI Thermal Spike Detected",
            "uid": f"ALERT-{int(datetime.now().timestamp())}",
            "desc": "Isolation Forest flagged abnormal industrial hardware thermal activity.",
            "analytic": {"name": "IsolationForest_Model", "type": "Anomaly Detection"}
        },
        "device": {"name": "Factory_Sensor_01", "type": "Industrial Sensor", "type_id": 1},
        "resources": [{"type": "Machine Metric", "name": "Temperature_Celsius", "value": str(round(test_temp, 2))}]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Normalized OCSF JSON Structure")
        st.json(ocsf_log)
        
    with col2:
        st.subheader("🤖 Google Gemma Co-Pilot Summary")
        with st.spinner("Gemma is decoding the OCSF schema..."):
            log_string = json.dumps(ocsf_log, indent=2)
            report = generate_gemma_report(log_string)
            st.markdown(report)
else:
    st.success(f"🟢 System Telemetry Stable: Current value ({test_temp}°C) operates safely within standard deviation limits.")
