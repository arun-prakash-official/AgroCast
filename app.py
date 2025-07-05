import streamlit as st
from utils import predict_image, generate_voice_message

st.set_page_config(page_title="AgroCast", layout="centered")
st.title("🌾 AgroCast – Plant Disease Detection with Voice Output")
st.markdown("Upload a plant leaf image to detect disease and get audio treatment advice.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="📷 Uploaded Leaf", use_container_width=True)
    with st.spinner("Analyzing leaf..."):
        label, confidence = predict_image(uploaded_file)
        result_text = f"The plant is detected as **{label}** with **{confidence:.2f}%** confidence."
        st.success(result_text)

        voice_msg = f"The result is {label}. I am {confidence:.1f} percent confident."
        audio_path = generate_voice_message(voice_msg)
        st.audio(audio_path, format='audio/mp3')
