import os
import cv2
import dlib
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy.spatial import distance

# 加载dlib的模型文件
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 初始化dlib的模型
predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
detector = dlib.get_frontal_face_detector()


# 图像缩放
def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)


# 灰度转换
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 面部检测
def detect_faces(image):
    gray = convert_to_grayscale(image)
    faces = detector(gray)
    return faces


# 面部标注
def detect_landmarks(image, faces):
    landmarks = []
    gray = convert_to_grayscale(image)
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append(shape)
    return landmarks


# 提取人脸特征向量
def extract_feature_vector(image, landmarks):
    face_descriptors = []
    for shape in landmarks:
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        face_descriptors.append(face_descriptor)
    return face_descriptors


# 比较人脸特征向量
def compare_faces(known_face_descriptor, face_descriptor_to_check):
    return distance.euclidean(known_face_descriptor, face_descriptor_to_check)


# 评估模型
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


# 加载训练和测试数据
train_data = load_data("path_to_train_folder")
test_data = load_data("path_to_test_folder")

# 评估模型性能
accuracy, recall, f1 = evaluate(train_data, test_data)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
