import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model

def load_models():
    # Load CNN model
    model_cnn = load_model(r'D:\proj\weights\cnn_nt_1.h5')

    # Load YOLO models
    model = YOLO(r'D:\proj\weights\final1.pt')
    model1 = YOLO(r'D:\proj\weights\final2.pt')
    model2 = YOLO(r'D:\proj\weights\final3.pt')
    
    return model_cnn, model, model1, model2
