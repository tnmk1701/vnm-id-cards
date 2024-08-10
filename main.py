import os
import cv2
import csv
import argparse
from model_loader import load_models
from image_utils import detect_and_display, reduce_brightness
from text_recognition import recognize_text_from_image

def main(image_directory, csv_output_path):
    class_names = ['bottom_left', 'bottom_right', 'top_left', 'top_right']

    model_cnn, model, model1, model2 = load_models()

    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'id'])

        for filename in os.listdir(image_directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(image_directory, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error loading image {filename}.")
                    continue

                # image = reduce_brightness(image)
                # image = cv2.resize(image, (900, 600))
                # transformed_img = image

                transformed_img = detect_and_display(image, model, class_names)

                if transformed_img is None:
                    transformed_img = image

                recognized_text = recognize_text_from_image(model1, model2, model_cnn, transformed_img)

                print(f'Image: {filename}, ID: {recognized_text}')

                writer.writerow([filename, recognized_text])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ID Card Extraction')
    parser.add_argument('--image_dir', type=str, help='Path to the image directory', default='D:\\vn-id-cards-extraction-project\\dataset\\test2')
    parser.add_argument('--csv_output', type=str, help='Path to the output CSV file', default='D:\\vn-id-cards-extraction-project\\id_te2_nt.csv')
    args = parser.parse_args()
    
    main(args.image_dir, args.csv_output)
