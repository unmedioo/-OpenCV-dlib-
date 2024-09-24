import cv2
import dlib

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
