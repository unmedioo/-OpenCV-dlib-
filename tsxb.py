import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  # 导入os模块


def extract_features(image_path):
    # 检查图像文件是否存在
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return

    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 如果图像为空，则返回
    if image is None:
        print("Failed to read image")
        return

    # 计算傅里叶变换
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # 计算功率谱密度
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift))

    # 显示原始图像和傅里叶变换后的图像
    plt.figure(figsize=(10, 6))
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# 使用函数，确保替换为正确的文件路径和文件名
extract_features('C:\\Users\\admin\\Pictures\\image identification\\2.jpg')
