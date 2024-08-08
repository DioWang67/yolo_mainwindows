import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import re

# 定義 CNN 模型
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 圖像預處理
def preprocess_image(image, image_size=(64, 64)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, image_size)
    return resized_image.reshape((64, 64, 1)) / 255.0

# 從文件名提取標籤
def extract_label(filename):
    match = re.search(r'J\w+', filename)
    if match:
        label = match.group(0)
        return label
    return None

# 加載數據
def load_data(data_dir):
    data = []
    targets = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            label = extract_label(filename)
            if label and label not in labels:
                labels.append(label)
            if label:
                image_path = os.path.join(data_dir, filename)
                image = cv2.imread(image_path)
                data.append(preprocess_image(image))
                targets.append(labels.index(label))
    return np.array(data), to_categorical(targets, num_classes=len(labels)), labels

# 加載新數據
def load_new_data(new_data_dir, existing_labels):
    new_data = []
    new_targets = []
    new_labels = existing_labels.copy()
    for filename in os.listdir(new_data_dir):
        if filename.endswith(".jpg"):
            label = extract_label(filename)
            if label:
                if label not in new_labels:
                    new_labels.append(label)
                image_path = os.path.join(new_data_dir, filename)
                image = cv2.imread(image_path)
                new_data.append(preprocess_image(image))
                new_targets.append(new_labels.index(label))
    return np.array(new_data), to_categorical(new_targets, num_classes=len(new_labels)), new_labels

if __name__ == "__main__":
    # 現有數據的路徑設置
    existing_data_dir = r'path_to_existing_data'  # 這是現有數據的目錄
    new_data_dir = r'path_to_new_samples'  # 新樣本的目錄

    # 加載現有數據和標籤
    X, y, labels = load_data(existing_data_dir)

    # 加載新數據
    new_X, new_y, new_labels = load_new_data(new_data_dir, labels)

    # 合併數據
    X_combined = np.concatenate((X, new_X), axis=0)
    y_combined = np.concatenate((y, new_y), axis=0)

    # 數據分割
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # 數據增強
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    datagen.fit(X_train)

    # 加載或建立模型
    model_path = 'cnn_model.keras'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        input_shape = (64, 64, 1)
        num_classes = len(new_labels)
        model = build_model(input_shape, num_classes)

    # 添加學習率調整和早停回調函數
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # 訓練模型
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=50, validation_data=(X_test, y_test),
                        callbacks=[lr_reduction, early_stopping])

    # 保存更新後的模型和標籤
    model.save('cnn_model_updated.keras')
    np.save('labels_updated.npy', new_labels)

    # 評估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # 顯示訓練過程
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    
    print(f"總圖片數量: {len(X_combined)}")
    print(f"訓練集大小: {len(X_train)}")
    print(f"測試集大小: {len(X_test)}")
    print(f"標籤數量: {len(new_labels)}")
    print(f"標籤: {new_labels}")
