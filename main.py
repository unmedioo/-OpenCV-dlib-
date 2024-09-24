import cv2
import dlib
import os
import shutil

# 加载预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 初始化人脸检测器
detector = dlib.get_frontal_face_detector()

# 指定大文件夹路径
root_folder = r"C:\Users\admin\Pictures\image identification"

# 创建输出文件夹
output_folder = r"C:\Users\admin\Pictures\image identification\processed_faces"
os.makedirs(output_folder, exist_ok=True)


# 递归遍历大文件夹下的所有文件夹和图片
def process_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".bmp"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                # 将图像转换为灰度图
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 使用dlib检测图像中的所有人脸
                faces = detector(gray_img)

                # 如果检测到人脸，绘制人脸矩形框和关键点
                if len(faces) > 0:
                    for face in faces:
                        shape = predictor(gray_img, face)

                        # 绘制人脸矩形框
                        x1 = max(face.left(), 0)
                        y1 = max(face.top(), 0)
                        x2 = min(face.right(), img.shape[1])
                        y2 = min(face.bottom(), img.shape[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 绘制人脸关键点
                        for point in shape.parts():
                            cv2.circle(img, (point.x, point.y), 2, (0, 0, 255), -1)

                    # 保存处理后的图片
                    output_filename = f"{file}"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, img)
                else:
                    # 如果没有检测到人脸，直接移动到'unrecognized_faces'文件夹
                    shutil.move(img_path, root + "\\unrecognized_faces")


# 开始处理
process_images(root_folder)

print("Processing completed.")
