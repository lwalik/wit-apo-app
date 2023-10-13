import tkinter as tk
from tkinter import filedialog
from classes.image_display import ImageDisplay
import cv2

class AppWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Aplikacja główna")
        self.geometry("300x200")

        open_button = tk.Button(self, text="Wczytaj obraz", command=self.load_and_display_image)
        open_button.pack(pady=50)

    def load_and_display_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki obrazów", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        if file_path:
            image = cv2.imread(file_path)
            ImageDisplay(image)
