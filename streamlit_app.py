import streamlit as st
import pandas as pd
import numpy as np

st.title("Machine Learning App")
st.info("This app will predict your obesity level!")

dff = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv") 
df = pd.DataFrame(dff)

# Expander to show raw data
with st.expander("Data"):
    st.write("This is a raw data")
    st.dataframe(df)
