import tkinter as tk
from tkinter import filedialog
from displays.image_display import ImageDisplay
from displays.histogram_display import HistogramDisplay

class AppWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ustawienie Okna
        self.title("Aplikacja OpenCV")
        self.geometry("800x600")
        self.configure(bg="lightgray")

        self.load_image_callback = None
        self.image_display = None
        self.histogram_display = None
        self.file_path = None

        # Dodanie przycisku do wczytania obrazu
        self.load_button = tk.Button(self, text="Wczytaj obraz", command=self.on_load_image_button_click)
        self.load_button.pack(pady=20)



    def get_image_path(selfself):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki obrazów", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        return file_path

    def set_load_image_callback(self, callback):
        self.load_image_callback = callback

    def on_load_image_button_click(self):
        self.file_path = self.get_image_path()
        if self.file_path:
            if self.image_display:
                self.image_display.destroy()  # Usuń poprzedni widok obrazu

            self.image_display = ImageDisplay(self, image_path=self.file_path)
            self.image_display.pack()  # Wyświetl obraz w tym samym oknie

            # Dodanie przycisku do stworzenia histogramu
            histogram_button = tk.Button(self, text="Create Histogram", command=self.create_histogram)
            histogram_button.pack()

    def create_histogram(self):
        if self.image_display:
            if self.histogram_display:
                self.histogram_display.destroy()
            self.histogram_display = HistogramDisplay(self,image_path=self.file_path)
            self.histogram_display.pack()

    def add_element_to_view(self, element):
        element.pack()