

import os
import json
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity
# try:
from module.cop3 import Picture
# except:
#     from cop3 import Picture

import math
from PyQt5.QtCore import QThread, pyqtSignal
from threading import Lock

import concurrent.futures
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class Image_similarity_checker:

    def __init__(self):
        self.picture = Picture()
        self.lock = Lock()
    
    def load_target_json(self, target_json_path):
        with open(target_json_path, "r") as json_file:
            target_data = json.load(json_file)
        return target_data

    def process_json_file(self, model_name, json_file_path, image_path, result_file):
        local_score_dir = []
        try:
            target_data = self.load_target_json(json_file_path)
            score = 0

            for item_name, item_data in target_data.items():
                region_coordinates = (
                    item_data["region_coordinates"]["x"],
                    item_data["region_coordinates"]["y"],
                    item_data["region_coordinates"]["width"],
                    item_data["region_coordinates"]["height"]
                )
                image_processing = (
                    item_data["image_processing"]["scale_factor"],
                    item_data["image_processing"]["block_size"],
                    item_data["image_processing"]["c"],
                    item_data["image_processing"]["kernel_size"]
                )
                ccd = item_data['image_name']
                print("####################################################")
                print(ccd)
                similarity_score, result = self.picture.process_model_file(
                    model_name, ccd, image_path, region_coordinates,
                    image_processing, result_file, os.path.dirname(json_file_path)
                )

                score = similarity_score
                if result == "Pass":
                    break
                print(score, result)
                print("####################################################")

            local_score_dir.append((ccd, score, result))
        except Exception as e:
            logging.error(f"未知错误：{e} 文件: {json_file_path} 项目: {item_name}")
            local_score_dir.append((ccd, "0.0", "Fail"))
        
        return local_score_dir

    def main(self, model_name, target_json_path, image_path, result_file):
        score_dir = []
        max_workers = 4  # Number of threads

        files = [os.path.join(root, file)
                 for root, _, files in os.walk(target_json_path)
                 for file in files if file.endswith('.json')]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_json_file, model_name, json_file, image_path, result_file) for json_file in files]

            for future in as_completed(futures):
                result = future.result()
                if result:
                    with self.lock:
                        score_dir.extend(result)

        return score_dir
    

    