import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import json
from datetime import datetime
from google import genai  # Official Google GenAI SDK

# Set page title and layout
st.set_page_config(page_title="AI Industrial Multi-Sensor Dashboard", layout="wide")

st.title("🏭 AI Industrial Multi-Sensor Dashboard (OCSF & Google Gemma)")
st.write("This dashboard leverages an unsupervised **Isolation Forest** to monitor both **Temperature and Pressure** simultaneously, auto-formats alerts into **OCSF JSON**, and drafts emergency response logs using **Gemma**.")

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
        Review this standardized OCSF (Open Cybersecurity Schema Framework) Detection Finding log tracking both temperature and pressure metrics:
        {ocsf_log_json}
        
        Write a brief, professional incident mitigation overview for factory operators.
        Your response must include:
        1. **Summary**: What happened based on the Temperature and Pressure metrics? Highlight if it's a dual-metric anomaly.
        2. **Risk Assessment**: How this specific combo of anomalies impacts production safety.
        3. **Mitigation Blueprint**: 2 steps the local engineering team should take immediately to vent pressure or cool the system.
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
            
            1. **Summary**: The Isolation Forest model detected severe operational deviations. The telemetry indicates pressure and temperature metrics have breached standard safety limits.
            2. **Risk Assessment**: High thermal activity paired with extreme fluid pressure introduces severe rupture risks, risking catastrophic structural failure of the transport line.
            3. **Mitigation Blueprint**:
               * **Step 1**: Actuate the primary pressure relief valve (PRV-01) to vent built-up line pressure immediately.
               * **Step 2**: Shut down the primary heater core and isolate the pressure delivery manifold.
            """
        return f"Could not connect to Gemma engine: {str(e)}"

# ----------------- DATA GENERATION & MODEL TRAINING -----------------
@st.cache_resource
def train_model():
    timestamps = pd.date_range(start="2026-06-11 00:00", periods=1000, freq="min")
    np.random.seed(27)
    
    # 1. Generate stable baselines
    # Temperature (Normal tightly grouped around 25°C)
    temperature = 25 + np.random.normal(0, 0.8, size=1000)
    # Pressure (Normal tightly grouped around 10.0 Bar)
    pressure = 10.0 + np.random.normal(0, 0.5, size=1000)
    
    # 2. Inject realistic anomalies (Spikes & Drops)
    # Hot Temp + High Pressure Spikes
    temperature[150:155] = 42.5
    pressure[150:155] = 18.2
    
    # Extreme Pressure Spike Only
    pressure[500:506] = 22.1
    
    # Cold Temp + Low Pressure Drops
    temperature[850:853] = 12.0
    pressure[850:853] = 3.1
    
    df_history = pd.DataFrame({
        "Timestamp": timestamps, 
        "Temperature_Celsius": temperature,
        "Pressure_Bar": pressure
    })
    
    # Train the Isolation Forest on BOTH features simultaneously for 2D classification
    model = IsolationForest(contamination=0.04, random_state=42)
    model.fit(df_history[['Temperature_Celsius', 'Pressure_Bar']])
    df_history['AI_Guess'] = model.predict(df_history[['Temperature_Celsius', 'Pressure_Bar']])
    
    return model, df_history

# Initialize our variables at startup so they are always defined!
ai_brain, df = train_model()

# ----------------- SECTION 1: HISTORICAL DATA VIEW -----------------
st.header("Historical Multi-Sensor Analysis")

# Double safety check: run if variables exist
if ai_brain is not None and df is not None:
    anomalies = df[df['AI_Guess'] == -1]

    # Create two subplots side-by-side sharing the time scale axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Subplot 1: Temperature Metrics
    ax1.plot(df['Timestamp'], df['Temperature_Celsius'], color='black', label='Normal Temp', alpha=0.4)
    ax1.scatter(anomalies['Timestamp'], anomalies['Temperature_Celsius'], color='red', label='AI Flagged Anomaly', zorder=5)
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.set_title("Historical AI Multi-Sensor Logs")

    # Subplot 2: Pressure Metrics
    ax2.plot(df['Timestamp'], df['Pressure_Bar'], color='blue', label='Normal Pressure', alpha=0.4)
    ax2.scatter(anomalies['Timestamp'], anomalies['Pressure_Bar'], color='red', label='AI Flagged Anomaly', zorder=5)
    ax2.set_ylabel("Pressure (Bar)")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    
    st.pyplot(fig)
else:
    st.warning("🔄 Training data is initializing. Please refresh the page in a moment.")

# ----------------- SECTION 2: LIVE SIMULATION -----------------
st.markdown("---")
st.header("⚡ Live Testing & OCSF Log Generator")
st.write("Drag the sliders to alter physical equipment telemetry values in real-time.")

col_slide1, col_slide2 = st.columns(2)
with col_slide1:
    test_temp = st.slider("Current Machine Sensor Value (°C):", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
with col_slide2:
    test_pressure = st.slider("Current Machine Pressure Value (Bar):", min_value=0.0, max_value=30.0, value=10.0, step=0.1)

# Ensure ai_brain is loaded before making a prediction
if ai_brain is not None:
    # 1. Run the AI prediction to see if the overall system detects an anomaly
    guess = ai_brain.predict([[test_temp, test_pressure]])[0]
    
    # 2. Define our hard thresholds to determine WHICH metric is causing the issue
    # Normal temp baseline is ~25°C, normal pressure is ~10 Bar
    is_temp_abnormal = test_temp < 20.0 or test_temp > 32.0
    is_pressure_abnormal = test_pressure < 8.0 or test_pressure > 14.0

    st.markdown("### 📊 Live Component Status")
    col_status1, col_status2 = st.columns(2)
    
    # --- TEMPERATURE INDEPENDENT ALERTS ---
    with col_status1:
        if is_temp_abnormal:
            st.error(f"❌ Temperature is Abnormal: {test_temp}°C")
        else:
            st.success(f"🟢 Temperature is Normal: {test_temp}°C")
            
    # --- PRESSURE INDEPENDENT ALERTS ---
    with col_status2:
        if is_pressure_abnormal:
            st.error(f"❌ Pressure is Abnormal: {test_pressure} Bar")
        else:
            st.success(f"🟢 Pressure is Normal: {test_pressure} Bar")

    # 3. If either metric is broken, the AI triggers the backend OCSF log generation
    if guess == -1 or is_temp_abnormal or is_pressure_abnormal:
        # Determine appropriate severity tier based on the split conditions
        if is_temp_abnormal and is_pressure_abnormal:
            severity_label = "Critical"
            severity_num = 5
            finding_title = "Multivariate Sensor Anomaly Detected"
        else:
            severity_label = "Major"
            severity_num = 4
            finding_title = f"Single-Metric {'Temperature' if is_temp_abnormal else 'Pressure'} Deviation"

        # Calculate local timestamp adjusted for India Standard Time (IST)
        local_timestamp = int(datetime.now().timestamp()) + 19800
        
        ocsf_log = {
            "metadata": {
                "version": "1.3.0",
                "product": {"vendor": "FactoryAI", "name": "Multi-Sensor Anomaly Engine", "version": "1.0.0"}
            },
            "time": local_timestamp,
            "class_uid": 2004,
            "class_name": "Detection Finding",
            "category_uid": 2,
            "category_name": "Findings",
            "activity_id": 1, 
            "severity_id": severity_num, 
            "severity": severity_label,
            "finding_info": {
                "title": finding_title,
                "uid": f"ALERT-{local_timestamp}",
                "desc": "Isolation Forest flagged abnormal industrial hardware activity.",
                "analytic": {"name": "IsolationForest_2D_Model", "type": "Unsupervised Multi-Variable"}
            },
            "device": {"name": "Factory_Sensor_01", "type": "Industrial Sensor Unit", "type_id": 1},
            "resources": [
                {"type": "Machine Metric", "name": "Temperature_Celsius", "value": str(round(test_temp, 2))},
                {"type": "Machine Metric", "name": "Pressure_Bar", "value": str(round(test_pressure, 2))}
            ]
        }
        
        st.markdown("---")
        col_json, col_copilot = st.columns(2)
        
        with col_json:
            st.subheader("📋 Normalized OCSF JSON Structure")
            st.json(ocsf_log)
            
        with col_copilot:
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
                        
                        Answer the user's question accurately. Use the current temperature ({test_temp}°C) and pressure ({test_pressure} Bar) anomalies to provide specific context. Keep it concise.
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
                                st.info("🤖 **Co-Pilot (Offline Mode):** For combined thermal and high-pressure threats, operators require Class 2 high-temp insulative gloves, safety goggles, a clear blast shield, and steel-toe boots before approaching `Factory_Sensor_01`.")
                            elif "valve" in q_lower or "coolant" in q_lower or "cause" in q_lower:
                                st.info("🤖 **Co-Pilot (Offline Mode):** This abnormal operational profile typically points to a mechanical valve failure, pressure regulator lockup, or a sudden line blockage downstream.")
                            else:
                                st.info(f"🤖 **Co-Pilot (Offline Mode):** The live AI engine is currently on a brief rate-limit cooldown. Regarding your query ('{user_question}'): Standard operating safety protocols dictate isolating the fluid line, closing the inlet manifold feed, and engaging the secondary loop radiator pumps.")
                        else:
                            st.error(f"Chat Error: {str(e)}")
                else:
                    st.warning("Please configure your API Key to use the interactive chat box.")
    else:
        st.success(f"🟢 System Telemetry Stable: Temperature ({test_temp}°C) and Pressure ({test_pressure} Bar) operate safely within standard deviation limits.")
else:
    st.info("🔄 Running training sequence. Please interact with the slider bars again.")
