"""
Created on Wed Jan 17 10:00:57 2024
@author:  Fahrettin Kuran
Title:    Turkiye-Specific Ground Motion Model for Cumulative Absolute Velocity (CAV) (KTP23)
Paper:    Kuran, F., Tanırcan, G., Pashaei, E. (2023) “Performance evaluation of 
          machine learning techniques in predicting cumulative absolute velocity”, 
          Soil Dynamics and Earthquake Engineering. 174, 108175. 
          https://doi.org/10.1016/j.soildyn.2023.108175.
Streamlit app
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

with open("ktp23_model.pkl", "rb") as file:
    model = pickle.load(file)
    
primaryColor="#426EDA"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#31333F"
font="Sans serif"

col1, col2 = st.columns([8, 2])
col1.subheader("Turkiye-Specific Ground Motion Model for Cumulative absolute velocity (CAV) (KTP23)")
col2.image("Kandilli_logo.png", use_column_width=True)

text = """
<strong>Gradient Boosting</strong> algorithm is implemented to predict the geomean <strong>cumulative absolute velocity (CAV)</strong> of two horizontal components.
A Turkiye-based regional ground motion model is obtained using <strong>The New Turkish Strong Motion Database (N-TSMD)</strong>.
"""
line_height = "1.2"
font_size = "14px"
st.markdown(f"<p style='line-height: {line_height}; font-size: {font_size};'>{text}</p>", unsafe_allow_html=True)

st.subheader("Estimator parameters")
input_info = [
    "<strong>Mw:</strong> Moment Magnitude",
    "<strong>Vs30:</strong> Shear wave velocity of the top 30 m of the soil (m/s)",
    "<strong>Rjb:</strong> Joyner-Boore distance  (km)",
    "<strong>SoF:</strong> Style-of-faulting"
]

line_height = "0.4"
font_size = "14px"
for line in input_info:
    st.markdown(f"<p style='line-height: {line_height}; font-size: {font_size};'>{line}</p>", unsafe_allow_html=True)
    

def user_input_features():
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 16px;'>Define estimator parameters</h1>", unsafe_allow_html=True)
    Mw = st.sidebar.slider("Mw", 3.5, 7.6, step=0.1)
    VS30 = st.sidebar.slider("Vs30", 131, 1862)
    Rjb = st.sidebar.slider("Rjb", 0, 200)
    SOF = st.sidebar.selectbox("SoF", ["Strike-slip", "Normal", "Reverse"])


    sof_encoding = {'Strike-slip': 0.52097948, 'Normal': 0.4475182, 'Reverse': 0.03150232}


    SOF_encoded = sof_encoding.get(SOF, 0)


    normalized_inputs = np.array([Mw, VS30, Rjb, SOF_encoded])


    min_values = np.array([3.5000, 131, 0.2919655592323920000, 0])
    max_values = np.array([7.6000, 1862, 199.40012782305000, 1])

    normalized_inputs = (normalized_inputs - min_values) / (max_values - min_values)

    actual_values = np.array([Mw, VS30, Rjb, SOF])

    return normalized_inputs.reshape(1, -1), actual_values

input_features, actual_values = user_input_features()

st.subheader("Summary of your inputs")

input_df = pd.DataFrame({'Estimator parameters': ['Mw', 'Vs30', 'Rjb', 'SoF'],
                          'Value': actual_values})


input_df_transposed = input_df.T


st.write(input_df_transposed)

if st.button("Submit"):
    prediction = model.predict(input_features)
    predicted_cav = max(0, prediction[0])

    st.subheader("Result")
    st.write(f"CAV = {predicted_cav:,.5f} cm/s")

paper = """
<strong>Publication:</strong> Kuran, F., Tanırcan, G., Pashaei, E. (2023) “Performance evaluation of machine learning techniques in predicting cumulative absolute velocity”, Soil Dynamics and Earthquake Engineering. 174, 108175. https://doi.org/10.1016/j.soildyn.2023.108175.
"""
line_height = "1.2"
font_size = "14px"
st.markdown(f"<p style='line-height: {line_height}; font-size: {font_size};'>{paper}</p>", unsafe_allow_html=True)
