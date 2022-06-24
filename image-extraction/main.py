import numpy as np
import cv2 as cv


def load_image():
    return cv.imread("./in.jpg")


def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def gaussian_blur(img):
    return cv.GaussianBlur(img, (5, 5), 0)


def otsu_binarization(img):
    ret3, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return img


def save(img):
    cv.imwrite("./out.jpg", img)


def dilate(img):
    iterations = 1
    kernel = np.ones((5, 5), np.uint8)
    return cv.dilate(img, kernel, iterations)


def find_contours(img):
    return cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


def find_bounding_boxes(contours):
    return [cv.boundingRect(c) for c in contours]


def save_subimages(original_image, boxes):
    i = 0
    for x, y, w, h in boxes:
        if w > 100 and h > 100:
            cropped = original_image[y : y + h, x : x + w]
            cv.imwrite(f"./results/ROI_{i}.jpg", cropped)
            i += 1


def main():
    original_image = load_image()

    img = grayscale(original_image)
    img = gaussian_blur(img)
    img = otsu_binarization(img)
    img = dilate(img)
    contours, hierarchy = find_contours(img)
    boxes = find_bounding_boxes(contours)
    save_subimages(original_image, boxes)


if __name__ == "__main__":
    main()
