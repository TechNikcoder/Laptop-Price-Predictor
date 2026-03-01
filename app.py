import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', sorted(df['Company'].unique()))

# Type of laptop
type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop (kg)', min_value=0.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
     '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)

# CPU
cpu = st.selectbox('CPU', sorted(df['Cpu brand'].unique()))

# Storage
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', sorted(df['Gpu brand'].unique()))

# OS
os_name = st.selectbox('OS', sorted(df['os'].unique()))

if st.button('Predict Price'):
    # Convert Yes/No to 1/0
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Compute PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = (((X_res ** 2) + (Y_res ** 2)) ** 0.5) / screen_size

    # Validate weight (optional but helpful)
    if weight <= 0:
        st.error("Please enter a valid laptop weight (greater than 0).")
    else:
        # Build input dataframe
        query_df = pd.DataFrame([{
            'Company': company,
            'TypeName': type_name,
            'Ram': int(ram),
            'Weight': float(weight),
            'Touchscreen': int(touchscreen_val),
            'Ips': int(ips_val),
            'ppi': float(ppi),
            'Cpu brand': cpu,
            'HDD': int(hdd),
            'SSD': int(ssd),
            'Gpu brand': gpu,
            'os': os_name
        }])

        # Predict (your model is trained on log(price), so exp back)
        pred = np.exp(pipe.predict(query_df)[0])

        st.success(f"The predicted price of this configuration is ₹ {int(pred)}")