# # app.py


import streamlit as st
from stt import speech_to_text

st.title("Speech to Text + Formality Detection")

uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("examples/temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("examples/temp.wav", format="audio/wav")
    st.write("Processing...")

    # Call function and display result
    result = speech_to_text("examples/temp.wav")
    st.success(result)   # Green box output
