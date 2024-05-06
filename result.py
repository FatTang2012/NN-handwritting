# result.py

from nn3_model import NN  # Đảm bảo rằng bạn đã tạo một file nn_model.py và chứa lớp Predictor trong đó
import numpy as np
import cv2


saved_model_path = 'save_model/my_model.npz'  # Đường dẫn tới file mô hình đã lưu
model = NN(X_train, y_train)  # Khởi tạo một instance của lớp NN
model.load_model(saved_model_path)  # Tải mô hình đã lưu

# Dự đoán kết quả cho dữ liệu mới
image_path = "D:\\Document move here\\Learning\\Hoc ki\\N2\\HK2 N2\\mang than kinh\\full set number\\ve.png"  # Đường dẫn đến ảnh mới
predicted_digit = model.digit_recognizer(image_path)  # Dự đoán chữ số từ ảnh mới
print("Predicted digit:", predicted_digit)  # In ra kết quả dự đoán