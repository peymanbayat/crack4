import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the trained model
model = None

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/model.h5')

def predict(image):
    # Preprocess the image
    resized_image = image.resize((224, 224))  # Resize the image to match the input size of the model
    normalized_image = tf.keras.preprocessing.image.img_to_array(resized_image) / 255.0  # Normalize the image
    input_image = tf.expand_dims(normalized_image, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(input_image)
    predicted_class_index = tf.argmax(prediction[0]).numpy()
    predicted_class_probability = prediction[0][predicted_class_index]
    
    return predicted_class_index, predicted_class_probability

# Define class labels
class_labels = ['Non-Defective', 'Defective']

st.header("Terminal Crimp Cross-section Classifier")

# Upload an image file
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "gif","bmp"])

# Check if an image file was uploaded
if uploaded_file is not None:
    # Load the model if it's not loaded
    if model is None:
        model = load_model()
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform prediction
    if st.button('Predict'):
        predicted_class_index, predicted_class_probability = predict(image)
        
        # Apply threshold
        threshold = 0.5
        if predicted_class_probability >= threshold:
            predicted_class = class_labels[predicted_class_index]
            st.success("Prediction: Non-Defective",icon="✅")
        else:
            predicted_class = 'Defective'
            st.error("Prediction: Defective ",icon="🚨")

        # Apply threshold
        #threshold = 0.5
        #if predicted_class_probability >= threshold:
            #predicted_class = class_labels[predicted_class_index]
        #else:
         #   predicted_class = 'Defective'
        
        # Display the prediction result
        #st.write("Prediction: ", predicted_class)
