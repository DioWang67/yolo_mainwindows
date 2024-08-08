import os
import json
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QComboBox, QPushButton, QLineEdit, QLabel


class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 400, 300)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.models_groupbox = QGroupBox("Models")
        self.models_layout = QVBoxLayout()
        self.models_groupbox.setLayout(self.models_layout)
        self.layout.addWidget(self.models_groupbox)

        self.model_input_layout = QVBoxLayout()
        self.model_input_edit = QLineEdit()
        self.add_model_button = QPushButton("Add Model")
        self.add_model_button.clicked.connect(self.add_model)
        self.model_input_layout.addWidget(self.model_input_edit)
        self.model_input_layout.addWidget(self.add_model_button)
        self.layout.addLayout(self.model_input_layout)

        self.model_dropdown = QComboBox()
        self.model_dropdown.currentIndexChanged.connect(self.update_model_dropdown)
        self.layout.addWidget(self.model_dropdown)

        self.config_file = "config.json"
        self.load_config()
        self.update_model_dropdown()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
                self.models_list = config.get("models", [])
        else:
            self.models_list = []

    def save_config(self):
        config = {"models": self.models_list}
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=4)

    def add_model(self):
        model_name = self.model_input_edit.text().strip()
        if model_name:
            model_widget = QWidget()
            model_layout = QHBoxLayout()
            model_widget.setLayout(model_layout)
            model_label = QLabel(model_name)
            delete_button = QPushButton("X")
            delete_button.clicked.connect(lambda: self.delete_model(model_widget, model_name))
            model_layout.addWidget(model_label)
            model_layout.addWidget(delete_button)
            self.models_layout.addWidget(model_widget)
            self.models_list.append(model_name)
            self.model_input_edit.clear()
            self.update_model_dropdown()
            self.save_config()

    def get_model_settings(self):
        return self.model_dropdown.currentText()

    def delete_model(self, widget, model_name):
        self.models_layout.removeWidget(widget)
        widget.deleteLater()
        self.models_list.remove(model_name)
        self.update_model_dropdown()
        self.save_config()

    def update_model_dropdown(self):
        self.model_dropdown.clear()
        self.model_dropdown.addItems(self.models_list)


def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SettingsWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
