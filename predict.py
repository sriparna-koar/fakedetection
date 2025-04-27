import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('deepfake_detection_model.h5')

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

class_indices = {'training_fake': 0, 'training_real': 1}
inv_class_indices = {v: k for k, v in class_indices.items()}

def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_label = np.argmax(prediction, axis=1)[0]
    
    predicted_class = inv_class_indices[class_label]
    
    if 'fake' in predicted_class:
        return "Fake"
    else:
        return "Real"
# Example usage
image_path = "real_and_fake_face_detection/real_and_fake_face/training_fake/easy_6_1110.jpg"
result = predict_image(image_path)
print(f"The image is {result}")
# Example usage
image_path = "real_and_fake_face_detection/real_and_fake_face/training_real/real_00001.jpg"
result = predict_image(image_path)
print(f"The image is {result}")