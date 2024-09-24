import cv2
import dlib
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score


# 假设的函数实现
def detect_faces(img):
    # 这里应该是你的人脸检测逻辑
    pass


def detect_landmarks(img, faces):
    # 这里应该是你的人脸关键点检测逻辑
    pass


def extract_feature_vector(img, landmarks):
    # 这里应该是你的特征向量提取逻辑
    pass


# 加载预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 初始化人脸检测器
detector = dlib.get_frontal_face_detector()

# 指定图像路径
image_path = r"C:\Users\admin\Pictures\image identification\3.jpg"

# 读取图像
img = cv2.imread(image_path)

# 检查图像是否成功加载
if img is None:
    print(f"Error loading image from path: {image_path}")
else:
    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用dlib检测图像中的所有人脸
    faces = detector(gray_img)

    # 判断是否检测到人脸
    if len(faces) == 0:
        print("未识别到有效人脸")
    else:
        # 遍历检测到的每个人脸
        for i, face in enumerate(faces):
            # 提取人脸区域
            x1 = face.left() if face.left() >= 0 else 0
            y1 = face.top() if face.top() >= 0 else 0
            x2 = face.right() if face.right() <= img.shape[1] else img.shape[1]
            y2 = face.bottom() if face.bottom() <= img.shape[0] else img.shape[0]

            # 绘制人脸矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 使用预测器提取人脸关键点
            shape = predictor(gray_img, face)
            for j, point in enumerate(shape.parts()):
                # 绘制人脸关键点
                cv2.circle(img, (point.x, point.y), 2, (0, 0, 255), -1)

        # 显示结果
        cv2.imshow("Faces Detected", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 评估模块
def evaluate(known_faces, test_faces):
    y_true = [face['label'] for face in test_faces]
    y_pred = []
    for test_face in test_faces:
        distances = [compare_faces(known_face['descriptor'], test_face['descriptor']) for known_face in known_faces]
        min_distance_index = distances.index(min(distances))
        y_pred.append(known_faces[min_distance_index]['label'])

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, recall, f1


def load_data(folder_path):
    data = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for file in os.listdir(label_folder):
                if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".bmp"):
                    img_path = os.path.join(label_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        faces = detect_faces(img)
                        if len(faces) > 0:
                            landmarks = detect_landmarks(img, faces)
                            descriptors = extract_feature_vector(img, landmarks)
                            for descriptor in descriptors:
                                data.append({"descriptor": descriptor, "label": label})
    return data


# 示例：加载数据并评估
folder_path = r"C:\Users\admin\Pictures\image identification"
data = load_data(folder_path)
# 假设有已知的known_faces数据
# accuracy, recall, f1 = evaluate(known_faces, data)
# 打印评估结果
# print(f"Accuracy: {accuracy}, Recall: {recall}, F1 Score: {f1}")
