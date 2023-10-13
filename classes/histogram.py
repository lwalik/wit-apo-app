import cv2
import numpy as np

class Histogram:
    def __init__(self, image):
        self.image = image

    def generate_histogram(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist_image = np.zeros((300, 256, 3), dtype=np.uint8)
        cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        for i in range(256):
            cv2.line(hist_image, (i, 300), (i, 300 - int(hist[i] * 300 / 255)), (255, 255, 255), 1)
        return hist_image
