import os
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import pandas as pd

class ImageClassifier:
    def __init__(self):
        """
        Initialize the ImageClassifier with a trained CNN model and label mapping.
        """
        model_path = './cnn_model.keras'
        labels_path = './labels.npy'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.model = load_model(model_path)
        self.labels = np.load(labels_path, allow_pickle=True)

    def preprocess_and_extract_edges(self, image, image_size=(64, 64)):
        """
        Convert an image to grayscale, resize it, and preprocess for CNN.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, image_size)
        reshaped_image = resized_image.reshape((64, 64, 1)) / 255.0  # Ensure data is in the range [0, 1]
        return reshaped_image

    def predict(self, image):
        """
        Predict the label and confidence score for a new image using CNN.
        """
        new_image_features = self.preprocess_and_extract_edges(image).reshape((1, 64, 64, 1))
        prediction = self.model.predict(new_image_features)
        max_proba_index = np.argmax(prediction)
        predicted_label = str(self.labels[max_proba_index])
        max_proba_score = np.float64(prediction[0][max_proba_index])
        
        return predicted_label, predicted_label, max_proba_score

    def extract_field_from_filename(self, filename):
        """
        Extract the field from a filename.
        """
        parts = filename.split('-')
        if len(parts) > 1:
            return parts[1]  # 提取文件名中的第二部分作為欄位值
        return None

    def compare_with_ccd_field(self, image, ccd_filename):
        """
        Predict the label of the image and compare it with the field extracted from the CCD filename.
        """
        predicted_label, max_proba_label, max_proba_score = self.predict(image)
        ccd_field = self.extract_field_from_filename(ccd_filename)
        match = predicted_label == ccd_field
        if match:
            # print(f"Type of ccd_field: {type(ccd_field)}")
            # print(f"Type of max_proba_label: {type(max_proba_label)}")
            # print(f"Type of max_proba_score: {type(max_proba_score)}")
            return ccd_field, max_proba_label, max_proba_score
        
        else:
            return ccd_field, max_proba_label, 0.0

# Example usage
if __name__ == "__main__":
    image_path = r'D:\Git\Talus_AOI0731CNN\module\target\A41402237001S-J2-1.jpg'  # 替換為您要測試的圖片路徑
    ccd_filename = "A41402237001S-J2-1.jpg"  # CCD 文件名
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    classifier = ImageClassifier()
    predicted_label, max_proba_label, max_proba_score = classifier.compare_with_ccd_field(image, ccd_filename)
    
    print(f"The predicted label for the new image is: {predicted_label}")
    print(f"Highest confidence score is for class {max_proba_label}: {max_proba_score:.2f}")
