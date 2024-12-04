import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from mtcnn import MTCNN
from PIL import Image, ImageDraw

# Load the trained model
model = load_model('/Users/yifan_/Desktop/Classes/CV/celebrity_classification_model3.h5')

# Define the class labels
class_labels = ['Angelina Jolie',
 'Brad Pitt',
 'Denzel Washington',
 'Hugh Jackman',
 'Jennifer Lawrence',
 'Johnny Depp',
 'Kate Winslet',
 'Leonardo DiCaprio',
 'Megan Fox',
 'Natalie Portman',
 'Nicole Kidman',
 'Robert Downey Jr',
 'Sandra Bullock',
 'Scarlett Johansson',
 'Tom Cruise',
 'Tom Hanks',
 'Will Smith']

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    """Resize and preprocess the uploaded image."""
    if image.mode != "RGB":  # Ensure the image is in RGB format
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize the image
    return image_array

# Detect and extract faces
def extract_faces(image):
    """Detect faces in the image and return cropped faces."""
    if image.mode != "RGB":  # Ensure the image is in RGB format
        image = image.convert("RGB")
    detector = MTCNN()
    image_array = np.array(image)
    detections = detector.detect_faces(image_array)
    faces = []
    for detection in detections:
        x, y, width, height = detection['box']
        x, y = max(0, x), max(0, y)
        cropped_face = image.crop((x, y, x + width, y + height))
        faces.append((cropped_face, (x, y, x + width, y + height)))
    return faces


# Streamlit app
st.set_page_config(page_title="Celebrity Face Classifier", layout="centered", page_icon="🎭")

# App title and description
st.title("🎭 Celebrity Face Classification")
st.markdown(
    """
    Welcome to the Celebrity Face Classifier! 🤩
    Upload an image of a celebrity, and our model will predict who they are.
    """
)


# Upload image section
uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("📸 Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect faces and classify
    if st.button("✨ Detect & Classify Faces"):
        with st.spinner("Processing... Please wait!"):
            faces = extract_faces(image)
            
            if len(faces) == 0:
                st.warning("No faces detected. Please try another image.")
            else:
                st.subheader("Detected Faces")
                for i, (face, bbox) in enumerate(faces):
                    st.image(face, caption=f"Face {i + 1}", use_column_width=True)
                    
                    # Preprocess and predict
                    processed_face = preprocess_image(face)
                    predictions = model.predict(processed_face)
                    predicted_class = class_labels[np.argmax(predictions)]
                    confidence = np.max(predictions) * 100

                    # Display prediction results
                    st.subheader(f"Face {i + 1}: **{predicted_class}**")
                    st.markdown(f"**Confidence Level:** {confidence:.2f}%")
else:
    st.info("👆 Upload an image to get started!")

