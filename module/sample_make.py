from PIL import Image, ImageEnhance
import random
import os

# 原始圖像的路徑
image_path = r'D:\Git\Talus_AOI0723\module\target\A41402237001S-J1-4.jpg'
# 打開原始圖像
original_image = Image.open(image_path)

# 創建保存增強圖像的目錄
output_dir = 'augmented_images'
os.makedirs(output_dir, exist_ok=True)

# 提取文件名前綴
base_name = os.path.basename(image_path)  # 獲取文件名
name_prefix = os.path.splitext(base_name)[0]  # 去除文件擴展名，得到前綴

# 設置增強圖像的數量
num_augmented_images = 500

for i in range(1, num_augmented_images + 1):  # 從1開始命名
    a = i+500
    # 隨機旋轉
    angle = random.uniform(-30, 30)
    augmented_image = original_image.rotate(angle)

    # 隨機縮放
    scale_factor = random.uniform(0.7, 1.3)
    new_size = (int(augmented_image.width * scale_factor), int(augmented_image.height * scale_factor))
    augmented_image = augmented_image.resize(new_size, Image.LANCZOS)  # 使用LANCZOS替代ANTIALIAS

    # 隨機調整亮度
    enhancer = ImageEnhance.Brightness(augmented_image)
    augmented_image = enhancer.enhance(random.uniform(0.8, 1.2))

    # 隨機水平翻轉
    if random.choice([True, False]):
        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)

    # 保存增強的圖像，命名格式為 <前綴><編號>.jpg
    output_filename = f'{name_prefix[:-1]}{a}.jpg'
    augmented_image.save(os.path.join(output_dir, output_filename))
