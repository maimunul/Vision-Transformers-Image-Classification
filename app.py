import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pretrained model
model_path = "E:\Desktop\VisionClassification\pretrained_vit.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names (replace with your actual class names)
class_names = ["apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot", "cauliflower",
               "chilli pepper", "corn", "cucumber", "eggplant", "garlic", "ginger", "grapes", "jalepeno", "kiwi",
               "lemon", "lettuce", "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple", "pomegranate",
               "potato", "raddish", "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip",
               "watermelon"]

# Fruits and vegetables categories
fruits = ["apple", "banana", "grapes", "kiwi", "lemon", "mango", "orange", "paprika", "pear", "pomegranate",
          "pineapple", "tomato", "watermelon"]
vegetables = ["beetroot", "bell pepper", "cabbage", "capsicum", "carrot", "cauliflower", "chilli pepper", "corn",
              "cucumber", "eggplant", "garlic", "ginger", "jalepeno", "lettuce", "onion", "peas", "potato", "raddish",
              "soy beans", "spinach", "sweetcorn", "sweetpotato", "turnip"]

# Streamlit app
st.title("Image Classification App Using Vision Transformers")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(preprocessed_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

    predicted_class_name = class_names[predicted_class]
    actual_class_name = "Fruits" if predicted_class_name in fruits else "Vegetables"
    
    st.write(f"Predicted class: {predicted_class_name}")
    st.write(f"Actual class: {actual_class_name}")
    st.write(f"Probability: {probabilities[predicted_class]:.4f}")
