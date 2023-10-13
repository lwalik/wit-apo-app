import tkinter as tk
from tkinter import filedialog, Menu
from classes.image_display import ImageDisplay
import cv2

class AppWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Aplikacja główna")
        self.geometry("300x50")

        # Tworzenie paska menu
        menubar = Menu(self)
        self.config(menu=menubar)

        # Dodawanie opcji do paska menu
        lab1_menu = Menu(menubar, tearoff=0)
        lab1_menu.add_command(label="Wczytaj obraz", command=self.load_and_display_image)
        menubar.add_cascade(label="Lab 1", menu=lab1_menu)

    def load_and_display_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki obrazów", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        if file_path:
            image = cv2.imread(file_path)
            ImageDisplay(image)
