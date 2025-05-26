import streamlit as st
from streamlit_shap import st_shap
import shap
import pickle

st.set_page_config(layout="wide")
st.title("XAI for Citation Screening")

shaps_pkl = open("shaps.pkl", "rb")
shaps = pickle.load(shaps_pkl)

possible_topics = shaps.keys()

option = st.selectbox("Select topic title", tuple(possible_topics))

st_shap(shap.plots.text(shaps[option]), height=300)
