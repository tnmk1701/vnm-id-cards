import cv2
import argparse
from model_loader import load_models
from image_utils import detect_and_display
from text_recognition import recognize_text_from_image

def main(image_path):
    class_names = ['bottom_left', 'bottom_right', 'top_left', 'top_right']

    # Load models
    model_cnn, model, model1, model2 = load_models()

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image {image_path}.")
        return

    transformed_img = detect_and_display(image, model, class_names)

    if transformed_img is None:
        transformed_img = image

    recognized_text = recognize_text_from_image(model1, model2, model_cnn, transformed_img)

    print(f'ID from image {image_path}: {recognized_text}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ID Card Extraction')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    main(args.image_path)
