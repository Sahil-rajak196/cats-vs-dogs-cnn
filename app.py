import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="🐶 AI Pet Classifier",
    page_icon="🐾",
    layout="wide"
)

# ==================================
# CUSTOM CSS (MODERN UI)
# ==================================
st.markdown("""
<style>

/* Gradient Background */
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.15);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    color: white;
}

/* Title Style */
.title-text {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: white;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #f1f1f1;
    margin-bottom: 40px;
}

/* Button */
.stButton>button {
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
    color: white;
    font-size: 20px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_cat_25k_final.h5")

model = load_model()

# ==================================
# HEADER
# ==================================
st.markdown('<div class="title-text">🐶🐱 AI Pet Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let AI decide: Dog or Cat?</div>', unsafe_allow_html=True)

# ==================================
# MAIN LAYOUT
# ==================================
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📸 Upload Your Pet Image", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("### 🤖 Model Details")
    st.write("• Deep Learning CNN Model")
    st.write("• TensorFlow / Keras")
    st.write("• Accuracy: 85%+")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================
# PREDICTION SECTION
# ==================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🚀 Analyze with AI"):

        with st.spinner("AI is thinking..."):
            time.sleep(1)
            prediction = model.predict(img_array)[0][0]

        dog_prob = float(prediction)
        cat_prob = float(1 - prediction)

        st.markdown("## 📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("🐶 Dog Probability", f"{dog_prob*100:.2f}%")
            st.progress(dog_prob)

        with result_col2:
            st.metric("🐱 Cat Probability", f"{cat_prob*100:.2f}%")
            st.progress(cat_prob)

        if dog_prob > cat_prob:
            st.success("🎉 It's a Dog!")
        else:
            st.success("🎉 It's a Cat!")