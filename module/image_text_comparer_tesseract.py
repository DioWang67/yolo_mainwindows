

import pytesseract
from PIL import Image as PILImage
from PIL import ImageTk
import cv2
import re
import logging
import numpy as np
import json
import os
from tkinter import filedialog

logging.basicConfig(filename='ocr_errors.log', level=logging.ERROR)

class ImageTextComparer:
    def __init__(self):
        # self.tesseract_cmd = r'C:\Users\diowang\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        self.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        self.eng_alphanumeric_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def read_image_text(self, image,score_tesseract):
        try:

            binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(binary_image, config=self.eng_alphanumeric_config)
            conf_text = pytesseract.image_to_data(binary_image, config=self.eng_alphanumeric_config, output_type=pytesseract.Output.DICT)
            filtered_text = ''
            for text, conf in zip(conf_text['text'], conf_text['conf']):
                if int(conf) >= score_tesseract:
                    print(int(conf))
                    filtered_text += text.strip()
                    confidence_value =conf
            # confidence_value = conf_text['conf'][4] if conf_text['conf'] else 0
            print(confidence_value,filtered_text)
            if not filtered_text:
                filtered_text = "None"

            return filtered_text, confidence_value
        except Exception as e:
            logging.error(f"Error reading image: {e}")
            return None, 0

    def preprocess_image(self, image):
        result = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return result

    def rescale_image(self, image, scale_factor):
        try:
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            dim = (width, height)
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        except Exception as e:
            logging.error(f"Error rescaling image: {e}")
            return image
    def grayscale_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def adaptive_threshold(self, image, block_size, c):
        if block_size < 3 or block_size % 2 == 0:
            block_size = 11
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    def median_filter(self, image, kernel_size):
        if kernel_size < 3 or kernel_size % 2 == 0:
            kernel_size = 5
        return cv2.medianBlur(image, kernel_size)

    def update_image(self, image, image_processing):
        self.processed_image_resize = self.rescale_image(image, image_processing[0])
        processed_image = self.preprocess_image(self.processed_image_resize)
        
        processed_image = self.grayscale_image(processed_image)
        processed_image = self.adaptive_threshold(processed_image, image_processing[1], image_processing[2])
        processed_image = self.median_filter(processed_image, image_processing[3])

        return processed_image

    def compare_images(self, ccd, image_path1, image_path2,image_processing,score_tesseract):


        processed_image2 = self.update_image(image_path2, image_processing)

        # text1, score1 = self.read_image_text(processed_image1)
        text2, score2 = self.read_image_text(processed_image2,score_tesseract)
        
        ccd_no_ext = ccd.split('.')[0]
        parts = ccd_no_ext.split('-')
        result = parts[1]

        print( text2,result)
        valid_pattern = r'^J\d{1,2}$'
        # valid_pattern = r'^J\d{2}$'
        if text2 is None or not re.match(valid_pattern, result) or not re.match(valid_pattern, text2):
            return 0, "Fail"

        if text2.strip() == result.strip():
            return score2, "Pass"
        else:
            return 0, "Fail"

if __name__ == "__main__":
    comparer = ImageTextComparer()
    image ="J-J8-1.jpg"
    image_path1 = cv2.imread(image)
    image_path2 = cv2.imread(image)

    result = comparer.compare_images(image, image_path1, image_path1)
    print(result)
