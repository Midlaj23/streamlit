import pandas as pd
import streamlit as st


data = pd.read_csv('datasets/iris.csv')

selected_columns = st.multiselect(label="Select One Column",options=data.columns)

st.write(f"You selected: {selected_columns}")

if selected_columns:
    st.write(data[selected_columns])