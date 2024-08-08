# worker.py
from PyQt5.QtCore import QThread, pyqtSignal

class Worker(QThread):
    finished = pyqtSignal(list)

    def __init__(self, checker, model_name, target_json_path, image_path, result_file):
        super().__init__()
        self.checker = checker
        self.model_name = model_name
        self.target_json_path = target_json_path
        self.image_path = image_path
        self.result_file = result_file
        print(model_name, target_json_path, image_path, result_file)

    def run(self):
        try:
            result = self.checker.main(self.model_name, self.target_json_path, self.image_path, self.result_file)
        except Exception as e:
            # 捕捉其他類型的錯誤
            print("未知錯誤：", e)
            result = [( ('', '0.0', 'Fail'))]
        self.finished.emit(result)
