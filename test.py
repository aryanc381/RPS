import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model from .keras format
model = load_model("rock-paper-scissors-model.keras")

# Define the class map
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors"
}

# Map predicted label to class name
def mapper(val):
    return REV_CLASS_MAP[val]

# Path to the input image (Update with your image path)
input_image_path = r"C:\Users\conta\Desktop\rockPaperScissor\rock-paper-scissors\image_data\scissors\2.jpg" 

# Load and preprocess the image
img = cv2.imread(input_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))  # Resize to match the input size
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict the class
pred = model.predict(img)
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

# Output the prediction
print(f"Predicted move: {move_name}")
