from nn3_model import NN  # Import class NN from nn3_model.py
from tkinter import Tk, Label, PhotoImage, TOP  # Import necessary components from tkinter
from PIL import Image, ImageTk  # Import Image and ImageTk modules from PIL
import numpy as np  # Import numpy
import cv2  # Import OpenCV
from keras.datasets import mnist  # Import MNIST dataset

saved_model_path = 'save_model/my_model.npz'  # Path to the saved model file

def digit_recognizer(image_path):
    # Initialize an instance of the NN class
    model = NN(None, None)
    # Load the saved model
    model.load_model(saved_model_path)
    # Read and process the image, then predict the digit
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (28, 28))
    normalized_image = resized_image.reshape(-1, 1).astype('float32') / 255.
    # Predict the digit
    prediction = model.predict(normalized_image)
    return prediction

def show_popup(image_path):
    # Create a root window
    root = Tk()
    root.title("Kết quả dự đoán")
    
    # Read and resize the image
    img = Image.open(image_path)
    img = img.resize((300, 300))
    
    # Convert the image to a format compatible with PhotoImage
    photo = ImageTk.PhotoImage(img)
    
    # Display the image in the popup window
    Label(root, image=photo).pack(side=TOP)

    # Display the predicted digit
    predicted_digit = digit_recognizer(image_path)
    Label(root, text="Kết quả dự đoán: {}".format(predicted_digit)).pack(side=TOP)

    root.mainloop()

image_path = "D:\\Document move here\\Learning\\Hoc ki\\N2\\HK2 N2\\mang than kinh\\full set number\\ve.png"
show_popup(image_path)
