import streamlit as st

st.set_page_config(
    page_title="TopVecSim",
    layout="centered",
)

st.title("TopVecSim")
st.write(
    "A Proof of Concept to demonstrate the use of Redis Vector Similarity Search and Top2Vec."
)
st.write(
    "A Bokeh visualization is in-progress for a 2D cluster view of the nearest embeddings."
)
