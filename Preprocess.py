import csv
import cv2
import numpy as np
import os

def fillHole(image):
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(image, mask, (0, 0), 255)
    return image

def fill_small_region(image, max_area, fill_value):
    fill_contours = []
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) <= max_area:
            fill_contours.append(contour)
    cv2.fillPoly(image, fill_contours, fill_value)
    return image

def FillHole2(img):
    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seed_found = False
    for i in range(h):
        for j in range(w):
            if im_floodfill[i, j] == 0:
                seedPoint = (i, j)
                seed_found = True
                break
        if seed_found:
            break
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = img | im_floodfill_inv
    return im_out

def lung_segment(input_path, save_path):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    init_thresh = np.mean(image)
    _, dst = cv2.threshold(gray, init_thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    constant = cv2.copyMakeBorder(dst, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    image = fillHole(constant)
    image = cv2.bitwise_not(image)
    image = fill_small_region(image, 800, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = FillHole2(image)

    image = cv2.resize(image, (512, 512))
    img = cv2.bitwise_and(image, gray)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

def process_dataset(input_dir, output_dir, error_csv):
    os.makedirs(output_dir, exist_ok=True)
    with open(error_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CTpath', 'error'])

    for label in os.listdir(input_dir):
        lpath = os.path.join(input_dir, label)
        for date in os.listdir(lpath):
            dpath = os.path.join(lpath, date)
            for ct in os.listdir(dpath):
                ct_path = os.path.join(dpath, ct)
                try:
                    for image_name in os.listdir(ct_path):
                        image_path = os.path.join(ct_path, image_name)
                        output_path = os.path.join(output_dir, label, date, ct, image_name)
                        lung_segment(image_path, output_path)
                    print(f"Processed: {ct_path}")
                except Exception as e:
                    with open(error_csv, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([ct_path, e])

if __name__ == "__main__":
    input_dir = 'input_dir'
    output_dir = 'output_dir'
    error_csv = 'error_csv'
    process_dataset(input_dir, output_dir, error_csv)
