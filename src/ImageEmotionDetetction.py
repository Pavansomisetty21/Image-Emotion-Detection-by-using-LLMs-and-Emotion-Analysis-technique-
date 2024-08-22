import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd

# Load your pre-trained emotion classifier model
pipe_lr = joblib.load(open(r"C:\Users\pavan\Downloads\emotion_classifier_pipe_lr.pkl", "rb"))  #Emotion Classifier model

# Function to predict emotion from text description
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Streamlit app
st.title("Image Description and Emotion Detection")

# Input: Google Gemini API key
api_key = st.text_input("Enter your Google Gemini API key:", type="password")

# Ensure the API key is set
if api_key:
    # Set up Google Gemini API
    genai.configure(api_key=api_key)

    # ClientFactory class to manage API clients
    class ClientFactory:
        def __init__(self):
            self.clients = {}
        
        def register_client(self, name, client_class):
            self.clients[name] = client_class
        
        def create_client(self, name, **kwargs):
            client_class = self.clients.get(name)
            if client_class:
                return client_class(**kwargs)
            raise ValueError(f"Client '{name}' is not registered.")

    # Register and create the Google generative AI client
    client_factory = ClientFactory()
    client_factory.register_client('google', genai.GenerativeModel) 

    client_kwargs = {
        "model_name": "gemini-1.5-flash",
        "generation_config": {"temperature": 0.8},
        "system_instruction": None,
    }

    client = client_factory.create_client('google', **client_kwargs)

    # User content for image description
    user_content = """Describe this picture, landscape, buildings, country, settings, and art style if any dictated. 
                      Identify any signs and indicate what they may suggest."""

    # Input: Image URL
    image_url = st.text_input("Enter the image URL:") #https://fal.media/files/elephant/zVhronC-mL1yhgLj3ubBu.png     is example url

    if image_url:
        try:
            # Load and display the image from URL
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption='Image from URL', use_column_width=True)
            
            # Generate image description
            st.write("Generating description...")
            response = client.generate_content([user_content, image_url], stream=True)
            response.resolve()
            description = response.text
            st.write(f"Image description: {description}")
            
            # Predict emotion based on the description
            st.write("Predicting emotion from the description...")
            emotion = predict_emotions(description)
            st.write(f"Detected Emotion: {emotion}")
            
            # Show prediction probabilities
            probability = get_prediction_proba(description)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            st.success("Prediction Probability")
            st.bar_chart(proba_df_clean.set_index('emotions'))

        except Exception as e:
            st.error(f"Error loading image: {e}")
else:
    st.warning("Please enter your API key to proceed.")
