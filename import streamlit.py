import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import json
from datetime import datetime

# Set page title and layout
st.set_page_config(page_title="AI Industrial Anomaly Detector", layout="wide")

st.title("🏭 AI Industrial Anomaly Detector & SIEM Logger")
st.write("This dashboard simulates factory machine temperature data and uses an **Isolation Forest AI model** to detect abnormal spikes and generate security alerts.")

# ----------------- DATA GENERATION & MODEL TRAINING -----------------
@st.cache_resource
def train_model():
    # Generate mock history data
    timestamps = pd.date_range(start="2026-06-11 00:00", periods=1000, freq="min")
    np.random.seed(27)
    temperature = 25 + np.random.normal(0, 1.5, size=1000)
    
    # Inject historical anomalies
    temperature[150:155] = 82.5
    temperature[500:506] = 79.1
    temperature[850:853] = 85.0

    df_history = pd.DataFrame({"Timestamp": timestamps, "Temperature_Celsius": temperature})
    
    # Train the AI Brain
    model = IsolationForest(contamination=0.015, random_state=42)
    model.fit(df_history[['Temperature_Celsius']])
    df_history['AI_Guess'] = model.predict(df_history[['Temperature_Celsius']])
    
    return model, df_history

ai_brain, df = train_model()

# ----------------- SECTION 1: HISTORICAL DATA VIEW -----------------
st.header("📈 Historical Sensor Analysis")
st.write("Below are the historical logs used to train the AI. The red markers indicate spikes flagged as anomalies.")

# Create the plot
normal_data = df[df['AI_Guess'] == 1]
anomalies = df[df['AI_Guess'] == -1]

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(df['Timestamp'], df['Temperature_Celsius'], color='black', label='Normal Reading', alpha=0.4)
ax.scatter(anomalies['Timestamp'], anomalies['Temperature_Celsius'], color='red', label='AI Detected Anomaly', zorder=5)
ax.set_title("AI Sensor Anomaly Detection Results")
ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(True)

# Display the plot in Streamlit instead of plt.show()
st.pyplot(fig)

# ----------------- SECTION 2: LIVE SIMULATION -----------------
st.markdown("---")
st.header("⚡ Live Testing & SIEM Log Generator")
st.write("Adjust the temperature below to test how the AI responds in real-time.")

# User interactive inputs
test_temp = st.slider("Set Current Machine Temperature (°C):", min_value=15.0, max_value=100.0, value=25.0, step=0.1)

# Run prediction
guess = ai_brain.predict([[test_temp]])[0]

# Display beautiful status updates instead of standard console print statements
if guess == -1:
    st.error(f"⚠️ SIEM ALERT TRIGGERED! Strange Temperature Detected: {test_temp}°C")
    
    # Generate the JSON log
    siem_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_device": "Factory_Sensor_01",
        "alert_level": "CRITICAL",
        "metric": "Temperature",
        "current_value": round(test_temp, 2),
        "status": "ANOMALY_DETECTED",
        "description": "AI detected unexpected thermal spike. Potential machine failure or sensor tampering."
    }
    
    st.subheader("Generated Security JSON Log:")
    st.json(siem_log)
else:
    st.success(f"🟢 System Healthy: The current temperature ({test_temp}°C) is within safe operational limits.")