from image_handler import ImageHandler
from image_card import ImageCard
import tkinter as tk

class ImageDisplay(tk.Frame):
    def __init__(self, parent, image_path, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.handler = ImageHandler(image_path)
        self.display_image()

    def display_image(self):
        image = ImageCard(self.handler.image)
        image_card = image.get_display_image()
        self.parent.add_element_to_view(image_card)

