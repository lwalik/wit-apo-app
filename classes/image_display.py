import cv2
from tkinter import Toplevel, Label, Button
from PIL import Image, ImageTk
from classes.histogram import Histogram

class ImageDisplay:
    def __init__(self, image):
        self.image = image
        self.display_image()

    def display_image(self):
        self.window = Toplevel()
        converted_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(converted_image)
        image_tk = ImageTk.PhotoImage(image=image_pil)

        label = Label(self.window, image=image_tk)
        label.image = image_tk  # Przechowaj referencję do obrazu
        label.pack()

        histogram_button = Button(self.window, text="Stwórz Histogram", command=self.create_histogram)
        histogram_button.pack(pady=10)

    def create_histogram(self):
        histogram = Histogram(self.image)
        hist_image = histogram.generate_histogram()

        hist_image_pil = Image.fromarray(hist_image)
        hist_image_tk = ImageTk.PhotoImage(image=hist_image_pil)

        label = Label(self.window, image=hist_image_tk)
        label.image = hist_image_tk
        label.pack(pady=10)
