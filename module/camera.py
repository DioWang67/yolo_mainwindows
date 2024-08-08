import sys
import os
import cv2
import numpy as np
import json
from datetime import datetime
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QThread, QObject, pyqtSlot, pyqtSignal, QMutex
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QApplication, QMainWindow, QPushButton, QGraphicsView, QVBoxLayout, QStatusBar


class CameraThread(QThread):
    stop_signal = pyqtSignal()
    save_frame_signal = pyqtSignal(str)
    frame_saved_signal = pyqtSignal(str, str)
    frame_ready_signal = pyqtSignal(QPixmap)
    camera_initialized_signal = pyqtSignal(bool)
    start_timer_signal = pyqtSignal()
    stop_timer_signal = pyqtSignal()
    stop_update_signal = pyqtSignal()  # 新增的信號

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.camera = None
        self.save_path = ""
        self.result_filename = ""
        self.camera_open_status = ""
        self.set_path = False
        self.mutex = QMutex()
        self.scene = QGraphicsScene()

    def run(self):
        self.camera = Camera(self.ui, self.mutex)
        self.camera.moveToThread(self)
        self.camera.frame_ready.connect(self.update_graphics_view)
        self.camera.config_updated.connect(self.set_paths_and_notify)
        
        self.camera.start_camera()
        self.stop_signal.connect(self.camera.stop_camera)
        self.camera.initialization_complete.connect(self.on_camera_initialized)
        self.save_frame_signal.connect(self.camera.save_current_frame)
        self.start_timer_signal.connect(self.camera.start_timer)
        self.stop_timer_signal.connect(self.camera.stop_timer)
        self.stop_update_signal.connect(self.camera.stop_update)  # 連接信號
        
        self.exec_()

    def stop_camera(self):
        self.stop_signal.emit()

    def save_current_frame(self,current_datetime):
        self.save_frame_signal.emit(current_datetime)

    @pyqtSlot(str, str)
    def set_paths_and_notify(self, save_path, result_folder):
        self.save_path, self.result_filename = save_path, result_folder
        self.frame_saved_signal.emit(save_path, result_folder)
        self.set_path = True

    @pyqtSlot(bool)
    def on_camera_initialized(self, status):
        self.camera_open_status = status
        if self.camera_open_status !="":
            self.stop_update_signal.emit()  # 發送信號
        self.camera_initialized_signal.emit(status)

    @pyqtSlot(QPixmap)
    def update_graphics_view(self, pixmap):
        self.scene.clear()
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)
        pixmap_item.setScale(0.7)
        self.ui.graphicsView.setScene(self.scene)

    def get_paths_or_default(self):
        return self.save_path, self.result_filename

    def get_camera_status(self):
        return self.camera_open_status

class Camera(QObject):
    frame_ready = pyqtSignal(QPixmap)
    config_updated = pyqtSignal(str, str)
    initialization_complete = pyqtSignal(bool)
    start_timer_signal = pyqtSignal()
    stop_timer_signal = pyqtSignal()

    def __init__(self, ui, mutex):
        super().__init__()
        self.ui = ui
        self.cap = None
        self.timer = QTimer()
        self.initialization_status = QTimer()
        self.is_camera_released = False
        self.is_frame_available = False
        self.update_interval = 15
        self.initialization_status_interval = 3000
        self.mutex = mutex

        self.cameranum = self.load_config("cameranum")
        self.resultpath = self.load_config("resultpath")
        self.timer.timeout.connect(self.update_frame)
        self.initialization_status.timeout.connect(self.update_camera_status)
        self.start_timer_signal.connect(self.start_timer)
        self.stop_timer_signal.connect(self.stop_timer)
        self.try_open_camera()

    def start_camera(self):
        self.start_timer_signal.emit()
        self.initialization_status.start(self.initialization_status_interval)

    def try_open_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.cameranum)
            if not self.cap.isOpened():
                self.start_timer_signal.emit()
            else:
                self.setup_camera()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.start_timer_signal.emit()

    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 180)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 20)
        self.cap.set(cv2.CAP_PROP_SATURATION, 24)
        self.cap.set(cv2.CAP_PROP_HUE, 0)
        self.cap.set(cv2.CAP_PROP_SHARPNESS, 8)
        self.cap.set(cv2.CAP_PROP_GAMMA, 60)

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    @pyqtSlot()
    def stop_camera(self):
        self.stop_timer_signal.emit()
        if self.cap:
            self.cap.release()

    @pyqtSlot()
    def stop_timer(self):
        if self.initialization_status.isActive():
            self.initialization_status.stop()
        if self.timer.isActive():
            self.timer.stop()

    @pyqtSlot()
    def start_timer(self):
        if not self.initialization_status.isActive():
            self.initialization_status.start(self.initialization_status_interval)
        if not self.timer.isActive():
            self.timer.start(self.update_interval)

    @pyqtSlot()
    def stop_update(self):
        if self.initialization_status.isActive():
            self.initialization_status.stop()
    def load_config(self, item):
        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
                if item == "selected_item":
                    item = config.get(item)
                else:
                    item = config["main"].get(item, "N/A")
                return item
        except FileNotFoundError:
            self.initialization_complete.emit(False)
            return "N/A"

    def update_camera_status(self):
        print("Updating camera status")
        if not self.cap.isOpened():
            self.initialization_complete.emit(False)
        else:
            self.initialization_complete.emit(True)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.mutex.lock()
            try:
                self.is_frame_available = True
                frame = cv2.resize(frame, (1280, 720))
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w

                self.qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(self.qt_image)
            finally:
                self.mutex.unlock()
            self.frame_ready.emit(self.pixmap)

    def save_current_frame(self,current_datetime):
        if self.is_frame_available:
            # current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            selected_item = self.load_config("selected_item")
            result_folder = os.path.join(self.resultpath, current_datetime, selected_item, f"cameranum{self.cameranum}")
            original_pic_folder = os.path.join(result_folder, "original")
            os.makedirs(original_pic_folder, exist_ok=True)
            save_path = os.path.join(original_pic_folder, f"{current_datetime}.jpg")

            image = self.pixmap.toImage()
            buffer = image.bits().asarray(image.byteCount())
            img_np = np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), image.width(), image.depth() // 8)
            cv2.imwrite(save_path, img_np)

            self.is_frame_available = False
            self.config_updated.emit(save_path, result_folder)
        else:
            print("No frame to save.")


