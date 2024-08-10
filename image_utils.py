import cv2
import numpy as np

def validate_coordinates(centers):
    tl = centers.get('top_left')
    tr = centers.get('top_right')
    bl = centers.get('bottom_left')
    br = centers.get('bottom_right')

    if tl and tr and bl and br:
        if (
            tl[0] < tr[0] and tl[0] < br[0] and tl[1] < bl[1] and tl[1] < br[1] and
            tr[0] > tl[0] and tr[0] > bl[0] and tr[1] < bl[1] and tr[1] < br[1] and
            bl[0] < tr[0] and bl[0] < br[0] and bl[1] > tl[1] and bl[1] > tr[1] and
            br[0] > tl[0] and br[0] > bl[0] and br[1] > tl[1] and br[1] > tr[1]
        ):
            print("Coordinates are valid.")
            return True
        else:
            print("Coordinates are invalid.")
            return False
    else:
        print("Not all corners are detected.")
        return False

# Detect and display function
def detect_and_display(image, yolo_model, class_names):
    original_image = image.copy()
    results = yolo_model.predict(image)
    centers = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()
            class_name = class_names[class_id]

            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(original_image, (center_x, center_y), 5, (255, 0, 0), -1)
            centers[class_name] = [center_x, center_y]

    if len(centers) == 4:
        if validate_coordinates(centers):
            source_points = np.float32([centers['bottom_left'], centers['bottom_right'], centers['top_left'], centers['top_right']])
            transformed_image = perspective_transform(image, source_points)
            return transformed_image
        else:
            print("Invalid coordinates. Returning the original image.")
            return image
    elif len(centers) == 3:
        centers = calculate_missed_coord_corner(centers)
        if validate_coordinates(centers):
            source_points = np.float32([centers['bottom_left'], centers['bottom_right'], centers['top_left'], centers['top_right']])
            transformed_image = perspective_transform(image, source_points)
            return transformed_image
        else:
            print("Invalid coordinates. Returning the original image.")
            return image
    else:
        print("Không phát hiện đủ 3 hoặc 4 class.")
        return image

def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 300], [500, 300], [0, 0], [500, 0]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst

def find_miss_corner(coordinate_dict):
    all_corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for i, corner in enumerate(all_corners):
        if corner not in coordinate_dict:
            return i, corner
    return None, None

def calculate_missed_coord_corner(coordinate_dict):
    thresh = 0
    _, missing_corner = find_miss_corner(coordinate_dict)

    if missing_corner == 'top_left':
        midpoint = np.add(coordinate_dict['top_right'], coordinate_dict['bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_right'][0] - thresh
        coordinate_dict['top_left'] = (x, y)
    elif missing_corner == 'top_right':
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_left'][0] - thresh
        coordinate_dict['top_right'] = (x, y)
    elif missing_corner == 'bottom_left':
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_right'][0] - thresh
        coordinate_dict['bottom_left'] = (x, y)
    elif missing_corner == 'bottom_right':
        midpoint = np.add(coordinate_dict['bottom_left'], coordinate_dict['top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_left'][0] - thresh
        coordinate_dict['bottom_right'] = (x, y)

    return coordinate_dict

def reduce_brightness(image, factor=0.9):
    if factor <= 0 or factor >= 1:
        raise ValueError("Factor must be in the range (0, 1).")
    image_float = image.astype(np.float32)
    image_darkened = image_float * factor
    image_darkened = np.clip(image_darkened, 0, 255).astype(np.uint8)
    return image_darkened

def convert2Square(image):
    h, w = image.shape[:2]
    if h > w:
        diff = h - w
        padding = diff // 2
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        diff = w - h
        padding = diff // 2
        padded_image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image
