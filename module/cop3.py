from skimage.metrics import structural_similarity
import cv2
import os
import json
from datetime import datetime
try :
    from module.image_text_comparer_tesseract import ImageTextComparer
except:
    from image_text_comparer_tesseract import ImageTextComparer
try:
    from module.CNN_predict_model import  ImageClassifier
except:
    from CNN_predict_model import  ImageClassifier

class Picture:
    def __init__(self):
        self.comparer = ImageTextComparer()
        self.classifier = ImageClassifier()
        self.goldenpath = None
        self.resultpath = None
        self.score_threshold = None

    def load_config(self):
        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
                self.goldenpath = config["main"]["goldenpath"]
                self.resultpath = config["main"]["resultpath"]
                self.score_threshold = config.get("score")
                self.score_tesseract = int(config.get("scoretesseract"))
        except FileNotFoundError:
            print("Configuration file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON configuration file.")

    def process_video_frame(self, picpath, ccd, region_coordinates, image_processing, result_file):
        try:
            a = cv2.imread(picpath)
            # 检查裁剪区域是否超出图像边界
            if region_coordinates[0] + region_coordinates[2] > a.shape[1] or region_coordinates[1] + region_coordinates[3] > a.shape[0]:
                raise ValueError("Crop area is out of image bounds.")
            choose_data = a[region_coordinates[1]:region_coordinates[1] + region_coordinates[3], region_coordinates[0]:region_coordinates[0] + region_coordinates[2]]
            original_picture = os.path.join(self.filepath, ccd)

            score, result = self.calculate_ssim(ccd, original_picture, choose_data, self.comparer, image_processing,self.score_tesseract)
            predicted_label, max_proba_label, max_proba_score = self.classifier.compare_with_ccd_field(choose_data,ccd)
            print(f"classifier={predicted_label, max_proba_label,max_proba_score}")

            score = score / 100
            CNN_score = round(max_proba_score, 2)
            if score > CNN_score:
                result_pic_name_score = score
            else:
                result_pic_name_score =CNN_score
            ccd_name = ccd.replace(".jpg", "")

            ccd_no_ext = ccd.split('.')[0]
            parts = ccd_no_ext.split('-')
            resultlabel = parts[1]
            
            if predicted_label == max_proba_label == resultlabel.strip():
                classifier_score = round(max_proba_score, 2)
            else :
                classifier_score = 0.0

            filename = os.path.join(result_file, f"{ccd_name}-{str(result_pic_name_score)}.jpg")

            cv2.imwrite(filename, choose_data)
            return score, result, classifier_score
        
        except Exception as e:
            print(f"Error processing video frame: {e}")
            return 0, "Fail", 0

    @staticmethod
    def calculate_ssim(ccd, image_path, compare_image, comparer, image_processing,score_tesseract):
        try:
            imageA = cv2.imread(image_path)
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)

            # 确保比较的两张图像具有相同的尺寸
            if grayA.shape != grayB.shape:
                raise ValueError("Input images must have the same dimensions.")

            score, result = comparer.compare_images(ccd, imageA, compare_image, image_processing,score_tesseract)

            # structural_similarity_score = str(round(structural_similarity_score, 2))
            return score, result
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0, "Fail"

    def process_model_file(self, model, ccd, picpath, region_coordinates, image_processing, result_file, target_json_path):
        self.load_config()
        self.filepath = target_json_path

        all_files = next(os.walk(self.filepath), (None, None, []))[2]
        if not all_files:
            print("No files found in the directory.")
            return "", "Fail"

        OCR_score, result, CNN_score = self.process_video_frame(picpath, ccd, region_coordinates, image_processing, result_file)
        print(type(OCR_score),type(CNN_score))
        float_score = float(OCR_score)
        CNN_score_float = float(CNN_score)
        print(OCR_score, result, CNN_score)

        if float_score > self.score_threshold and float_score > CNN_score_float:
            return str(float_score), result
        elif CNN_score_float > self.score_threshold and CNN_score_float > float_score:
            return str(CNN_score_float), "Pass"
        else:
            return "0.0", "Fail"

        # if float_score > self.score_threshold and CNN_score_float < float_score:
        #     return str_score, result
        # elif CNN_score>self.score_threshold and CNN_score_float> float_score:
        #     return str_structural_similarity_score, "Pass"
        # else:
        #     return "0.0","Fail"

if __name__ =="__main__" :
    P = Picture()
    model_name = "A41402237001S"

    image_path =r"D:\Git\Talus_AOI0731CNN\module\target\2.jpg"
    result_file = r"D:\Git\Talus_AOI0731CNN\module\target\result"
    json_file_path=r"D:\Git\Talus_AOI0731CNN\module\target\A41402237001S-J2.json"
    def load_target_json( target_json_path):
        with open(target_json_path, "r") as json_file:
            target_data = json.load(json_file)
        return target_data

    def process_json_file(model_name, json_file_path, image_path, result_file):
        local_score_dir = []

        target_data = load_target_json(json_file_path)
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



            similarity_score, result = P.process_model_file(
            model_name, ccd, image_path, region_coordinates,
            image_processing, result_file, os.path.dirname(json_file_path)
            )

            print(similarity_score, result)
            process_json_file(model_name, json_file_path, image_path, result_file)