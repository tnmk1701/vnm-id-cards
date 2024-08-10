import cv2
import numpy as np
from image_utils import convert2Square

def recognize_text_from_image(model1, model2, cnn_model, transformed_img):
    results = model1.predict(transformed_img)

    recognized_text = ''
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0])

            if class_id == 0:
                cropped_img = transformed_img[y1:y2, x1:x2]

                results2 = model2.predict(cropped_img)

                characters = []
                for res in results2:
                    for box in res.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        char_img = cropped_img[y1:y2, x1:x2]

                        gray_char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
                        _, binary_char_img = cv2.threshold(gray_char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        binary_char_img = cv2.bitwise_not(binary_char_img)
                        binary_char_img = cv2.medianBlur(binary_char_img, 5)

                        characters.append((binary_char_img, x1))

                characters = sorted(characters, key=lambda x: x[1])

                candidates = []
                for char_img, x in characters:
                    char_img = convert2Square(char_img)
                    char_img = cv2.resize(char_img, (28, 28), cv2.INTER_AREA)
                    char_img = char_img.reshape((28, 28, 1))
                    candidates.append((char_img, x))

                for candidate, x in candidates:
                    candidate = candidate.astype('float32') / 255.0
                    candidate = np.expand_dims(candidate, axis=0)
                    prediction = cnn_model.predict(candidate)
                    predicted_char = np.argmax(prediction, axis=1)
                    if predicted_char == 10:
                        recognized_text += "/"
                    else:
                        recognized_text += str(predicted_char[0])
    return recognized_text
