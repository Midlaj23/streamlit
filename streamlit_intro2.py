import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.title("North Pole Penguin")

penguins = pd.read_csv("datasets/penguins.csv.xls")
st.write(penguins)

st.selectbox