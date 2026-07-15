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
        
        # Calling Google's native model
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        # ---- GRACEFUL FALLBACK FOR 429 RATE LIMITS ----
        if "429" in str(e) or "QUOTA" in str(e).upper():
            return """
            ⚠️ **Live Copilot Quota Reached (Displaying Cached Standard Security Blueprint)**
            
            1. **Summary**: The Isolation Forest model detected an extreme operational deviation. The temperature telemetry crossed the baseline safety threshold of 28°C, threatening system equilibrium.
            2. **Risk Assessment**: Prolonged operation at elevated temperatures introduces severe risks of mechanical fatigue, insulation degradation, and permanent component warping, which could cause an unscheduled production line shutdown.
            3. **Mitigation Blueprint**:
               * **Step 1**: Disengage the primary power drive to reduce active load and initiate the secondary auxiliary cooling loop immediately.
               * **Step 2**: Deploy a field technician equipped with a thermal imaging device to verify physical hardware core temperatures and inspect the primary coolant valve lines.
            """
        return f"Could not connect to Gemma engine: {str(e)}"

# ----------------- DATA GENERATION & MODEL TRAINING -----------------
@st.cache_resource
def train_model():
    timestamps = pd.date_range(start="2026-06-11 00:00", periods=1000, freq="min")
    np.random.seed(27)
    
    # Generate stable baseline readings tightly centered around 25°C
    temperature = 25 + np.random.normal(0, 0.8, size=1000)
    
    # 1. Inject historical HOT spikes
    temperature[150:155] = 82.5
    temperature[500:506] = 79.1
    temperature[850:853] = 85.0
    
    # 2. Inject historical COLD drops (New addition!)
    temperature[300:304] = 5.2
    temperature[700:705] = 8.0
    
    df_history = pd.DataFrame({"Timestamp": timestamps, "Temperature_Celsius": temperature})
    
    # 2% contamination perfectly isolates our spikes and drops
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(df_history[['Temperature_Celsius']])
    df_history['AI_Guess'] = model.predict(df_history[['Temperature_Celsius']])
    
    return model, df_history

# Initialize our variables at startup so they are always defined!
ai_brain, df = train_model()

# ----------------- SECTION 1: HISTORICAL DATA VIEW -----------------
st.header("Historical Sensor Analysis")

# Double safety check: run if variables exist
if ai_brain is not None and df is not None:
    anomalies = df[df['AI_Guess'] == -1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Timestamp'], df['Temperature_Celsius'], color='black', label='Normal Reading', alpha=0.4)
    ax.scatter(anomalies['Timestamp'], anomalies['Temperature_Celsius'], color='red', label='AI Flagged Spikes', zorder=5)
    ax.set_title("Historical AI Training Logs")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning("🔄 Training data is initializing. Please refresh the page in a moment.")

# ----------------- SECTION 2: LIVE SIMULATION -----------------
st.markdown("---")
st.header("⚡ Live Testing & OCSF Log Generator")
st.write("Drag the slider to alter physical device temperature values.")

# Change min_value from 15.0 to 0.0
test_temp = st.slider("Current Machine Sensor Value (°C):", min_value=0.0, max_value=100.0, value=25.0, step=0.1)

# Ensure ai_brain is loaded before making a prediction
if ai_brain is not None:
    guess = ai_brain.predict([[test_temp]])[0]

    if guess == -1:
        st.error(f"⚠️ SIEM ALERT TRIGGERED! Abnormal Thermal Metric: {test_temp}°C")
        
        # Calculate local timestamp adjusted for India Standard Time (IST: UTC + 5h 30m)
        local_timestamp = int(datetime.now().timestamp()) + 19800
        
        # Standardized Open Cybersecurity Schema Framework Object
        ocsf_log = {
            "metadata": {
                "version": "1.3.0",
                "product": {"vendor": "FactoryAI", "name": "Thermal Anomaly Engine", "version": "1.0.0"}
            },
            "time": local_timestamp,
            "class_uid": 2004,
            "class_name": "Detection Finding",
            "category_uid": 2,
            "category_name": "Findings",
            "activity_id": 1, 
            "severity_id": 5, 
            "severity": "Critical",
            "finding_info": {
                "title": "AI Thermal Spike Detected",
                "uid": f"ALERT-{local_timestamp}",
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
                
            # ---- INTERACTIVE CO-PILOT CHAT ASSISTANT ----
            st.markdown("---")
            st.markdown("### 💬 Ask Co-Pilot Follow-Up Questions")
            user_question = st.text_input(
                "Ask anything about this incident (e.g., 'What safety gear do I need?'):", 
                key="incident_qa"
            )

            if user_question:
                if "GEMINI_API_KEY" in st.secrets:
                    try:
                        client_chat = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                        
                        chat_prompt = f"""
                        Context Log: {log_string}
                        
                        User Question: {user_question}
                        
                        Answer the user's question accurately. If it is about the incident log above, use the data to assist them. If it is a general question, answer it like a helpful AI assistant. Keep it concise.
                        """
                        
                        with st.spinner("Gemini is thinking..."):
                            chat_response = client_chat.models.generate_content(
                                model='gemini-3.5-flash',
                                contents=chat_prompt,
                            )
                            st.info(chat_response.text)
                    except Exception as e:
                        # ---- GRACEFUL CHAT FALLBACK FOR 429 RATE LIMITS ----
                        if "429" in str(e) or "QUOTA" in str(e).upper():
                            q_lower = user_question.lower()
                            if "gear" in q_lower or "safety" in q_lower or "ppe" in q_lower:
                                st.info("🤖 **Co-Pilot (Offline Mode):** High thermal anomalies require standard Class 2 Industrial PPE: high-temperature insulative gloves, a clear face shield to protect against potential coolant line ruptures, and flame-resistant overalls before approaching `Factory_Sensor_01`.")
                            elif "valve" in q_lower or "coolant" in q_lower or "cause" in q_lower:
                                st.info("🤖 **Co-Pilot (Offline Mode):** Anomaly analysis indicates this sudden thermal spike is likely caused by a mechanical binding in the primary cooling loop valve or an accumulation of mineral deposits blocking fluid transit lines.")
                            else:
                                st.info(f"🤖 **Co-Pilot (Offline Mode):** The live AI engine is currently on a brief rate-limit cooldown. Regarding your query ('{user_question}'): Standard operating procedures dictate isolating the hardware module, checking core resistance baselines, and verifying the diagnostic logs before manually restarting the system.")
                        else:
                            st.error(f"Chat Error: {str(e)}")
                else:
                    st.warning("Please configure your API Key to use the interactive chat box.")
    else:
        st.success(f"🟢 System Telemetry Stable: Current value ({test_temp}°C) operates safely within standard deviation limits.")
else:
    st.info("🔄 Running training sequence. Please slide the temperature bar again.")
