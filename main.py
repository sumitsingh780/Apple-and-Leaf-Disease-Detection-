import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Function to load the model
def load_model():
    try:
        model = tf.keras.models.load_model('fruit_leaf_classification_insception_20_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

# If model loaded successfully
if model:
    # Define class labels
    class_labels = ['Apple Scab', 'Black Rot', 'Blotch_Apple', 'Cedar Apple Rust', 'Healthy', 'Normal_Apple', 'Rot_Apple', 'Scab_Apple']

    # Function to preprocess and predict
    def predict_image(img):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        return class_labels[predicted_class]
    
    
     
    # Streamlit app configuration
    st.set_page_config(
      page_title="Fruit Leaf Classification",
      page_icon="🌳",  # Use an apple emoji as the favicon
      layout="centered"  # Expand the app to fill the browser width
    )
    

    # Add a header with a background image
    st.header('🍎Fruit Leaf Classification')
    st.write("Identify and diagnose your fruit leaves!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Classify image on button click
        if st.button('Classify 🔍'):
            # Preprocess and predict
            img = image.load_img(uploaded_file, target_size=(299, 299))
            label = predict_image(img)
            
            # Display prediction
            st.write(f"Prediction: {label}")
            st.write("**Note:** This is just an automated prediction. For further information, consult a plant specialist.")
