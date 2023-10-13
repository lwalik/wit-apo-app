import tkinter as tk
from services.image_service import ImageService
from classes.histogram import Histogram

class HistogramDisplay(tk.Frame):
    def __init__(self, parent, image_path, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.handler = ImageService(image_path)
        self.display_histogram()


    def display_histogram(self):
        histogram = Histogram(self.handler.image)
        histogram_image = histogram.get_histogram_image()
        self.parent.add_element_to_view(histogram_image)