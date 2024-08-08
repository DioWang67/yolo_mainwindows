
# target.py
import os
import json
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsPixmapItem, QFileDialog, QLabel, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import threading

class ImageSelectionWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_region = None
        self.setWindowTitle("Target make")

        # Initialize variables
        self.image = None
        self.image_path = ""
        self.start = None
        self.end = None
        self.is_selecting = False

        # Set up graphics view and scene
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Create UI elements
        self.load_button = QPushButton("Select Image")
        self.load_button.clicked.connect(self.load_image)
        self.save_button = QPushButton("Save Selected Region")
        self.save_button.clicked.connect(self.save_selected_region)
        self.optimize_button = QPushButton("Optimize Parameters")
        self.optimize_button.clicked.connect(self.optimize_parameters)
        self.ocr_button = QPushButton("Perform OCR")
        self.ocr_button.clicked.connect(self.perform_ocr)

        self.model_label = QLabel("機種:")
        self.model_edit = QLineEdit()
        self.item_label = QLabel("項目:")
        self.item_edit = QLineEdit()
        self.layer_label = QLabel("層數:")
        self.layer_edit = QLineEdit()
        self.camera_label = QLabel("相機:")
        self.camera_edit = QLineEdit()

        # New UI elements for image processing configuration
        self.scale_factor_label = QLabel("縮放係數:")
        self.scale_factor_edit = QLineEdit()
        self.block_size_label = QLabel("區塊大小:")
        self.block_size_edit = QLineEdit()
        self.c_label = QLabel("C值:")
        self.c_edit = QLineEdit()
        self.kernel_size_label = QLabel("內核大小:")
        self.kernel_size_edit = QLineEdit()

        # Layout for controls
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.model_label)
        controls_layout.addWidget(self.model_edit)
        controls_layout.addWidget(self.item_label)
        controls_layout.addWidget(self.item_edit)
        controls_layout.addWidget(self.layer_label)
        controls_layout.addWidget(self.layer_edit)
        controls_layout.addWidget(self.camera_label)
        controls_layout.addWidget(self.camera_edit)
        controls_layout.addWidget(self.scale_factor_label)
        controls_layout.addWidget(self.scale_factor_edit)
        controls_layout.addWidget(self.block_size_label)
        controls_layout.addWidget(self.block_size_edit)
        controls_layout.addWidget(self.c_label)
        controls_layout.addWidget(self.c_edit)
        controls_layout.addWidget(self.kernel_size_label)
        controls_layout.addWidget(self.kernel_size_edit)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.optimize_button)
        controls_layout.addWidget(self.ocr_button)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(controls_widget, 1)
        main_layout.addWidget(self.view, 3)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set window size
        self.resize(1280, 720)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)")
        if file_path:
            self.image_path = file_path
            self.image = QPixmap(self.image_path)
            self.scene.clear()
            self.scene.addPixmap(self.image)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport():
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                self.start = self.view.mapToScene(event.pos())
                self.end = None
                self.is_selecting = True
                self.selected_region = None
            elif event.type() == event.MouseMove and self.is_selecting:
                self.end = self.view.mapToScene(event.pos())
                self.update_selection()
            elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton and self.is_selecting:
                self.is_selecting = False
                self.save_selection()
        return super().eventFilter(obj, event)

    def save_selection(self):
        if self.start and self.end:
            self.selected_region = self.get_selection_rect()

    def update_selection(self):
        if not self.image:
            return
        pixmap = self.image.copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        if self.selected_region:
            painter.drawRect(self.selected_region)
        if self.start and self.end:
            rect = self.get_selection_rect()
            painter.drawRect(rect)
        painter.end()
        self.scene.clear()
        self.scene.addPixmap(pixmap)

    def get_selection_rect(self):
        if self.start and self.end:
            return QRectF(self.start, self.end).normalized()

    def save_selected_region(self):
        if not self.image:
            QMessageBox.warning(self, 'Error', '請選擇圖片。')
            return

        if not self.selected_region:
            QMessageBox.warning(self, 'Error', '請框選辨識範圍。')
            return

        self.model = self.model_edit.text()
        self.item = self.item_edit.text()
        self.layer = self.layer_edit.text()
        self.camera = self.camera_edit.text()

        # Retrieve image processing parameters
        scale_factor = self.scale_factor_edit.text()
        block_size = self.block_size_edit.text()
        c_value = self.c_edit.text()
        kernel_size = self.kernel_size_edit.text()

        if not all([self.model, self.item, self.layer, self.camera, scale_factor, block_size, c_value, kernel_size]):
            QMessageBox.warning(self, 'Error', '請輸入所有必要的資料。')
            return

        json_dir = os.path.join("target", "golden", f"{self.model}-data", f"camera{self.camera}")
        os.makedirs(json_dir, exist_ok=True)
        rect = self.selected_region.toRect()
        pixmap = self.image.copy(rect)
        image_name = f"{self.model}-{self.item}-{self.layer}.jpg"
        save_path = os.path.join(json_dir, image_name)
        pixmap.save(save_path)
        print(f"Saved region: {save_path}")
        print(f"Region coordinates: {rect}")

        region_data = {
            "x": rect.x(),
            "y": rect.y(),
            "width": rect.width(),
            "height": rect.height()
        }

        image_processing = {
            "scale_factor": float(scale_factor),
            "block_size": int(block_size),
            "c": int(c_value),
            "kernel_size": int(kernel_size)
        }

        json_path = os.path.join(json_dir, f"{self.model}-{self.item}.json")

        if os.path.isfile(json_path):
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
            data[image_name] = data.get(image_name, {"image_name": image_name})
            data[image_name]["region_coordinates"] = region_data
            data[image_name]["image_processing"] = image_processing
        else:
            data = {
                image_name: {
                    "image_name": image_name,
                    "region_coordinates": region_data,
                    "image_processing": image_processing
                }
            }

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Saved JSON file: {json_path}")

    def optimize_parameters(self):
        if not self.image:
            QMessageBox.warning(self, 'Error', '請選擇圖片。')
            return

        if not self.selected_region:
            QMessageBox.warning(self, 'Error', '請框選辨識範圍。')
            return
        
        item = self.item_edit.text()
        if not all([item]):
            QMessageBox.warning(self, 'Error', '請輸入項目名稱。')
            return
        
        rect = self.selected_region.toRect()
        selected_pixmap = self.image.copy(rect)
        selected_image = selected_pixmap.toImage()
        buffer = selected_image.bits().asstring(selected_image.byteCount())
        np_image = np.frombuffer(buffer, dtype=np.uint8).reshape((selected_image.height(), selected_image.width(), 4))
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGRA2BGR)

        best_score = 0
        best_params = None
        num_iterations = 100  # Number of random searches

        lock = threading.Lock()

        def search_parameters():
            nonlocal best_score, best_params
            for _ in range(num_iterations // 4):  # 分成4個線程
                sf = round(random.uniform(3.5, 5.0), 1)
                bs = random.randint(3, 57)
                if bs % 2 == 0:
                    bs += 1  # Ensure block_size is odd
                c = random.randint(3, 20)
                ks = random.randint(3, 20)
                if ks % 2 == 0:
                    ks += 1  # Ensure kernel_size is odd

                processed_image = self.process_image(np_image, sf, bs, c, ks)
                text, score = self.read_image_text(processed_image)
                print(text, score)
                with lock:
                    if score > best_score:
                        best_score = score
                        best_params = (sf, bs, c, ks)
                    if best_score >= 90:
                        print("END")
                        print(text, score, best_params)
                        return

        threads = []
        for _ in range(4):
            thread = threading.Thread(target=search_parameters)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if best_params:
            self.scale_factor_edit.setText(str(best_params[0]))
            self.block_size_edit.setText(str(best_params[1]))
            self.c_edit.setText(str(best_params[2]))
            self.kernel_size_edit.setText(str(best_params[3]))
            QMessageBox.information(self, 'Optimization Complete', f'Optimized parameters found with score: {best_score}')
        else:
            QMessageBox.warning(self, 'Optimization Failed', 'Failed to optimize parameters to achieve score above 90.')

    def process_image(self, image, scale_factor, block_size, c_value, kernel_size):
        # Resize image
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold
        block_size = max(3, block_size) if block_size % 2 == 1 else 11
        threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)

        # Apply median filter
        kernel_size = max(3, kernel_size) if kernel_size % 2 == 1 else 5
        filtered_image = cv2.medianBlur(threshold_image, kernel_size)

        return filtered_image

    def read_image_text(self, image):

        tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


        try:
            binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(binary_image, config=custom_config)
            conf_text = pytesseract.image_to_data(binary_image, config=custom_config, output_type=Output.DICT)
            
            filtered_text = ''.join([txt.strip() for txt, conf in zip(conf_text['text'], conf_text['conf']) if int(conf) >= 0])
            confidence_value = conf_text['conf'][4] if len(conf_text['conf']) > 4 else 0
            item = self.item_edit.text()
            if filtered_text == item.strip():
                return filtered_text or "None", confidence_value
            else:
                return None, 0
        except Exception as e:
        
            return None, 0

    def perform_ocr(self):
        if not self.image:
            QMessageBox.warning(self, 'Error', '請選擇圖片。')
            return

        if not self.selected_region:
            QMessageBox.warning(self, 'Error', '請框選辨識範圍。')
            return
        
        item = self.item_edit.text()
        if not all([item]):
            QMessageBox.warning(self, 'Error', '請輸入項目名稱。')
            return
        
        rect = self.selected_region.toRect()
        selected_pixmap = self.image.copy(rect)
        selected_image = selected_pixmap.toImage()
        buffer = selected_image.bits().asstring(selected_image.byteCount())
        np_image = np.frombuffer(buffer, dtype=np.uint8).reshape((selected_image.height(), selected_image.width(), 4))
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGRA2BGR)

        # Perform OCR
        processed_image = self.process_image(np_image, float(self.scale_factor_edit.text()), int(self.block_size_edit.text()), int(self.c_edit.text()), int(self.kernel_size_edit.text()))
        text, score = self.read_image_text(processed_image)

        QMessageBox.information(self, 'OCR Result', f'Recognized text: {text}\nConfidence score: {score}')

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = ImageSelectionWindow()
    window.show()
    sys.exit(app.exec_())
