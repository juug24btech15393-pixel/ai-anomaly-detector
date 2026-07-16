import streamlit as st
import random
import time
import json
from datetime import datetime

# =========================================================================
# SECTION 1: CORE ENGINE & MODEL INITIALIZATION
# =========================================================================

# Mocking the AI model loading logic
# In your production code, replace this with your actual loaded Isolation Forest or pipeline model
if "ai_brain" not in st.session_state:
    st.session_state.ai_brain = "Loaded Anomaly Engine v1.0"

ai_brain = st.session_state.ai_brain

def generate_gemma_report(log_json_str):
    """
    Simulates Google Gemma parsing the OCSF log and generating a 
    clean, human-readable industrial incident report summary.
    """
    try:
        log_data = json.loads(log_json_str)
        severity = log_data.get("severity", "Unknown")
        title = log_data.get("finding_info", {}).get("title", "Metric Deviation")
        metrics = log_data.get("resources", [])
        
        metric_summary = ", ".join([f"{m['type']} ({m['name']}) = {m['value']}" for m in metrics])
        
        report = f"""
        **🚨 GEMMA COPILOT AUTOMATED INCIDENT REPORT**
        
        * **Status:** Alert Generated via FactoryAI Engine
        * **Classification:** {title}
        * **Assigned Severity:** Level {log_data.get('severity_id', 3)} ({severity})
        
        **Analysis:**  
        The automated monitoring pipeline captured telemetry violating standard runtime boundaries. 
        Current metrics show: {metric_summary}. 
        
        **Recommended Action:**  
        Inspect physical valves or heating elements immediately. Validate Modbus registry loops to rule out transient sensor calibration errors.
        """
        return report
    except Exception as e:
        return f"Error parsing log data for Gemma report: {str(e)}"

# =========================================================================
# SECTION 2: STREAMLIT UI SETUP & LIVE SIMULATION
# =========================================================================

st.set_page_config(page_title="Industrial Anomaly Dashboard", layout="wide")

st.title("🏭 Automated Industrial Anomaly Detection Dashboard")
st.write("This application monitors real-time operational telemetry and structures anomalies into normalized OCSF logs.")

st.markdown("---")
st.header("⚡ Live Industrial Sensor Stream")
st.write("Toggle the live feed to simulate real-time Modbus/MQTT hardware data collection.")

# 1. Add an activation switch for the automatic stream
auto_stream = st.checkbox("📡 Connect to Live Machine Data Feed", value=False)

# Use Streamlit's session state to persist values across quick automatic reruns
if "sim_temp" not in st.session_state:
    st.session_state.sim_temp = 25.0
if "sim_pressure" not in st.session_state:
    st.session_state.sim_pressure = 10.0

if auto_stream:
    # 2. Simulate live industrial fluctuations
    # Introduce random minor ambient drift (normal behavior)
    temp_drift = random.uniform(-0.5, 0.5)
    press_drift = random.uniform(-0.2, 0.2)
    
    st.session_state.sim_temp = round(st.session_state.sim_temp + temp_drift, 1)
    st.session_state.sim_pressure = round(st.session_state.sim_pressure + press_drift, 1)
    
    # Keep normal values bounded so they don't drift to infinity
    st.session_state.sim_temp = max(22.0, min(st.session_state.sim_temp, 28.0))
    st.session_state.sim_pressure = max(9.0, min(st.session_state.sim_pressure, 11.0))
    
    # 3. Inject an occasional mechanical anomaly (10% chance) to test red/green logic
    if random.random() < 0.10:
        anomaly_type = random.choice(["high_pressure", "low_pressure", "dual_spike"])
        if anomaly_type == "high_pressure":
            st.session_state.sim_pressure = round(random.uniform(18.0, 24.0), 1)  # Mechanical blockage
        elif anomaly_type == "low_pressure":
            st.session_state.sim_pressure = round(random.uniform(2.0, 5.0), 1)    # System leak
        elif anomaly_type == "dual_spike":
            st.session_state.sim_temp = round(random.uniform(35.0, 45.0), 1)
            st.session_state.sim_pressure = round(random.uniform(20.0, 26.0), 1)

current_temp = st.session_state.sim_temp
current_pressure = st.session_state.sim_pressure

# Display static metric status info banner
st.info(f"📡 Data Feed Status: {'ACTIVE - Streaming Telemetry' if auto_stream else 'PAUSED'}")

# =========================================================================
# SECTION 3: METRIC EVALUATION & VISUAL ALERTS
# =========================================================================

# Evaluate alerts using the separate component threshold rules
is_temp_abnormal = current_temp < 20.0 or current_temp > 32.0
is_pressure_abnormal = current_pressure < 8.0 or current_pressure > 14.0

st.markdown("### 📊 Live Component Status")
col_status1, col_status2 = st.columns(2)

with col_status1:
    if is_temp_abnormal:
        st.error(f"❌ Temperature is Abnormal: {current_temp}°C")
    else:
        st.success(f"🟢 Temperature is Normal: {current_temp}°C")
        
with col_status2:
    if is_pressure_abnormal:
        st.error(f"❌ Pressure is Abnormal: {current_pressure} Bar")
    else:
        st.success(f"🟢 Pressure is Normal: {current_pressure} Bar")

# =========================================================================
# SECTION 4: OCSF LOGGING & GENMA AI CO-PILOT PIPELINE
# =========================================================================

# 5. Process OCSF and Gemma Logs if any anomaly occurs
if ai_brain is not None and (is_temp_abnormal or is_pressure_abnormal):
    if is_temp_abnormal and is_pressure_abnormal:
        severity_label, severity_num = "Critical", 5
        finding_title = "Multivariate Sensor Anomaly Detected"
    else:
        severity_label, severity_num = "Major", 4
        finding_title = f"Single-Metric {'Temperature' if is_temp_abnormal else 'Pressure'} Deviation"

    # Add localized timestamp offset if needed (e.g., IST +5:30 -> 19800 seconds)
    local_timestamp = int(datetime.now().timestamp())
    
    ocsf_log = {
        "metadata": {
            "version": "1.3.0",
            "product": {"vendor": "FactoryAI", "name": "Multi-Sensor Anomaly Engine", "version": "1.0.0"}
        },
        "time": local_timestamp,
        "class_uid": 2004,
        "class_name": "Detection Finding",
        "severity_id": severity_num, 
        "severity": severity_label,
        "finding_info": {"title": finding_title, "desc": "Automated pipeline captured out-of-bounds metrics."},
        "resources": [
            {"type": "Machine Metric", "name": "Temperature_Celsius", "value": str(current_temp)},
            {"type": "Machine Metric", "name": "Pressure_Bar", "value": str(current_pressure)}
        ]
    }
    
    st.markdown("---")
    col_json, col_copilot = st.columns(2)
    
    with col_json:
        st.subheader("📋 Normalized OCSF JSON")
        st.json(ocsf_log)
        
    with col_copilot:
        st.subheader("🤖 Google Gemma Co-Pilot Summary")
        report = generate_gemma_report(json.dumps(ocsf_log))
        st.markdown(report)

# =========================================================================
# SECTION 5: STREAMLIT AUTOMATIC RERUN LOOP
# =========================================================================

# 6. Infinite loop trigger to force Streamlit to refresh the UI automatically every second
if auto_stream:
    time.sleep(1)
    st.rerun()
