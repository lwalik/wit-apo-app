from image_handler import ImageHandler
import cv2
from PIL import Image, ImageTk
import tkinter as tk

class ImageCard():
    def __init__(self, image):
        self.image = image


    def get_display_image(self):
        image_cv = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_cv)
        image_tk = ImageTk.PhotoImage(image=image_pil)

        label = tk.Label(image=image_tk)
        label.image = image_tk
        return label