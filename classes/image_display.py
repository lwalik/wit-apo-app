import cv2
from tkinter import Toplevel, Label, Menu, filedialog
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

        # Tworzenie paska menu
        menubar = Menu(self.window)
        self.window.config(menu=menubar)

        # Dodawanie opcji do paska menu
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Zapisz Obraz", command=self.save_image)
        file_menu.add_command(label="Stwórz Histogram", command=self.create_histogram)
        file_menu.add_command(label="Duplikuj Obraz", command=self.duplicate_image)
        menubar.add_cascade(label="Lab 1", menu=file_menu)

        label = Label(self.window, image=image_tk)
        label.image = image_tk  # Przechowaj referencję do obrazu
        label.pack(pady=10)

    def create_histogram(self):
        histogram = Histogram(self.image)
        hist_image = histogram.generate_histogram()

        hist_image_pil = Image.fromarray(hist_image)
        hist_image_tk = ImageTk.PhotoImage(image=hist_image_pil)

        label = Label(self.window, image=hist_image_tk)
        label.image = hist_image_tk
        label.pack(pady=10)

    def duplicate_image(self):
        ImageDisplay(self.image)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.image)
