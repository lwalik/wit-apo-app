import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

class Histogram:
    def __init__(self, image):
        self.image = image

    def create_histogram_image(self):
        if self.image is None:
            raise ValueError("Obraz nie został wczytany.")

        # Konwertuj obraz na odcienie szarości, jeśli nie jest
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Oblicz histogram
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()

        # Tworzenie obrazu histogramu
        histogram_image = np.zeros((100, 256, 3), dtype=np.uint8)
        for i in range(256):
            cv2.line(histogram_image, (i, 100), (i, 100 - int(histogram[i] * 100)), (255, 255, 255), 1)

        return histogram_image

    def get_histogram_image(self):
        histogram_image = self.create_histogram_image()
        image_pil = Image.fromarray(cv2.cvtColor(histogram_image, cv2.COLOR_BGR2RGB))
        image_tk = ImageTk.PhotoImage(image=image_pil)
        label = tk.Label(image=image_tk)
        label.image = image_tk  # Przechowaj referencję do obrazu, aby uniknąć problemów z garbage collector'em
        return label
