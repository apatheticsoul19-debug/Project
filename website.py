import streamlit as st
import joblib
import pandas as pd
import time

# --- Page Config ---
st.set_page_config(page_title="Gemini Smart Grid", page_icon="💎", layout="wide")

# --- Memory Management ---
if 'my_appliances' not in st.session_state:
    st.session_state.my_appliances = ["Medical Monitor", "Refrigerator", "Air Conditioner"]
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Power"])


# --- Load the ML Brain ---
@st.cache_resource
def load_model():
    return joblib.load('power_model.pkl')


try:
    model = load_model()
except:
    st.error("Model file not found! Please run your training script first.")
    st.stop()


# --- The Smart Intel Engine ---
def get_device_intel(name):
    n = name.lower()
    intel_map = {
        "heart": (1, "❤️", "Critical", 50),
        "medical": (1, "🏥", "Critical", 100),
        "monitor": (1, "🖥️", "Critical", 60),
        "security": (1, "🛡️", "Critical", 40),
        "fridge": (2, "❄️", "Essential", 150),
        "refrigerator": (2, "🧊", "Essential", 200),
        "light": (2, "💡", "Essential", 15),
        "wifi": (2, "🌐", "Essential", 20),
        "fan": (2, "🌀", "Essential", 75),
        "ac": (3, "🌬️", "High Load", 1500),
        "conditioner": (3, "❄️", "High Load", 2000),
        "charger": (3, "🔌", "High Load", 3000),
        "tesla": (3, "🚗", "High Load", 7000),
        "ev": (3, "⚡", "High Load", 5000),
        "heater": (3, "🔥", "High Load", 1500),
        "oven": (3, "🥧", "High Load", 2000)
    }
    for key, value in intel_map.items():
        if key in n: return value
    return (3, "🔌", "General", 500)  # Default wattage and priority


# --- Sidebar: Controls ---
st.sidebar.header("📡 Live Sensor Data")
v_in = st.sidebar.slider("Voltage (V)", 200, 260, 230)
c_in = st.sidebar.number_input("Amperage (A)", 0.0, 30.0, 5.0)
total_w = v_in * c_in

st.sidebar.markdown("---")
st.sidebar.header("➕ Add Appliance")
new_item = st.sidebar.text_input("Device Name")
if st.sidebar.button("Add to Grid"):
    if new_item and new_item not in st.session_state.my_appliances:
        st.session_state.my_appliances.append(new_item)
        st.rerun()

# --- Main Dashboard ---
st.title("⚡ Smart Power Load Balancer")

# ML Prediction
pred = model.predict([[v_in, c_in]])[0]

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("Live Power Usage", f"{total_w:.1f} W")

# Calculate Potential Savings
saved_w = 0
for name in st.session_state.my_appliances:
    prio, _, _, watt_est = get_device_intel(name)
    if pred == "High" and prio == 3:
        saved_w += watt_est

m2.metric("Power Saved", f"{saved_w} W", delta=f"-{saved_w}" if saved_w > 0 else None)

if pred == "High":
    m3.error(f"STATUS: {pred} LOAD")
else:
    m3.success(f"STATUS: {pred} LOAD")

st.divider()

# --- Device Management Grid ---
cols = st.columns(4)
for i, name in enumerate(st.session_state.my_appliances):
    prio, emoji, cat, _ = get_device_intel(name)
    is_on = True if pred != "High" or prio < 3 else False

    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"### {emoji} {name}")
            st.caption(f"{cat} (P{prio})")
            if is_on:
                st.success("🟢 ACTIVE")
            else:
                st.error("🔴 SHEDDED")

            if st.button(f"Remove", key=f"btn_{i}"):
                st.session_state.my_appliances.remove(name)
                st.rerun()

# --- Power History Graph ---
st.divider()
st.subheader("📊 Power Consumption History")
new_data = pd.DataFrame({"Time": [time.time()], "Power": [total_w]})
st.session_state.history = pd.concat([st.session_state.history, new_data]).tail(20)
st.line_chart(st.session_state.history.set_index("Time"))



import streamlit as st
from PIL import Image
import os

# Get the path where this script is saved
script_directory = os.path.dirname(__file__)
image_path = os.path.join(script_directory, 'circuit_diagram.png')

# Display the image in the sidebar
if os.path.exists(image_path):
    img = Image.open(image_path)
    st.sidebar.image(img, caption="Virtual Hardware Model (Cirkit Design IDE)")
else:
    st.sidebar.error("Could not find 'circuit_diagram.png'. Check your project folder!")

