import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from gtts import gTTS
import os
from PIL import Image
import uuid

model = tf.keras.models.load_model("plant_disease_model.h5")
IMG_SIZE = 224

# Replace with your actual class labels
class_labels = ['Tomato___Healthy', 'Tomato___Early_blight', 'Tomato___Late_blight']

def predict_image(img_file):
    img = Image.open(img_file).resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

def generate_voice_message(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_file = f"temp/{uuid.uuid4()}.mp3"
    os.makedirs("temp", exist_ok=True)
    tts.save(audio_file)
    return audio_file
