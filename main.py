import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, Button, Label, Toplevel, Menu, BOTH, Frame, Canvas, Scrollbar
from PIL import Image, ImageTk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.figure import Figure

class ImageWindow:
    def __init__(self, image_path):
        self.top = Toplevel()
        self.top.title(os.path.basename(image_path))
        self.image = cv2.imread(image_path)
        self.display_image()
        Button(self.top, text="Stwórz Histogram", command=self.show_histogram).pack()
        self.show_lut_button = Button(self.top, text="Pokaż tablicę LUT", command=self.show_lut_table)
        self.show_lut_button.pack()
        self.lut_window = None  # Okno tablicy LUT
        self.lut_array = self.calculate_lut_array(self.image)  # Oblicz tablicę LUT

    def display_image(self):
        b, g, r = cv2.split(self.image)
        img = cv2.merge((r, g, b))
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label = Label(self.top, image=imgtk)
        label.imgtk = imgtk  # Keep a reference to the image object to prevent garbage collection
        label.pack()

    def show_histogram(self):
        histogram_window = Toplevel()
        histogram_window.title("Histogram - " + self.top.title())
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        image = self.image.copy()

        if len(image.shape) == 2:
            # Obraz monochromatyczny
            self.calculate_and_plot_histogram(image, ax, 'gray')
        elif len(image.shape) == 3:
            if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 0], image[:, :, 2]):
                # Wszystkie kanały są takie same, traktujemy obraz jako monochromatyczny
                self.calculate_and_plot_histogram(image[:, :, 0], ax, 'gray')
            else:
                # Obraz kolorowy
                colors = ('b', 'g', 'r')
                for i, color in enumerate(colors):
                    self.calculate_and_plot_histogram(image[:, :, i], ax, color)

        ax.set_xlabel("Wartość piksela")
        ax.set_ylabel("Liczba pikseli")
        ax.set_xlim([0, 256])

        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=BOTH, expand=True, side='top')

    def calculate_and_plot_histogram(self, channel, ax, color):
        hist = np.zeros(256)
        for value in range(256):
            hist[value] = np.sum(channel == value)
        ax.bar(range(256), hist, color=color, alpha=0.6)

    def calculate_lut_array(self, image):
        lut_array = np.zeros(256, dtype=np.uint32)  # Utwórz tablicę LUT o rozmiarze 256 komórek
        height, width = image.shape[:2]
        for i in range(height):
            for j in range(width):
                pixel_value = image[i, j, 0]  # Zakładamy, że obraz jest w skali szarości
                lut_array[pixel_value] += 1  # Zwiększ liczbę pikseli o wartości `pixel_value` o 1
        return lut_array

    def show_lut_table(self):
        if self.lut_window is None:
            self.lut_window = Toplevel()
            self.lut_window.title("Tablica LUT")
            self.lut_window.geometry("300x500")  # Ustaw rozmiar okna

            lut_canvas = Canvas(self.lut_window)
            lut_canvas.pack(fill=BOTH, expand=True)

            scrollbar = Scrollbar(lut_canvas, orient="vertical", command=lut_canvas.yview)
            scrollbar.pack(side="right", fill="y")
            lut_canvas.configure(yscrollcommand=scrollbar.set)

            lut_frame = Frame(lut_canvas)
            lut_canvas.create_window((0, 0), window=lut_frame, anchor="nw")

            for i in range(256):
                Label(lut_frame, text=f"{i}", width=5, anchor="w").grid(row=i, column=0)
                Label(lut_frame, text=f"{self.lut_array[i]}", width=10, anchor="w").grid(row=i, column=1)

            lut_frame.update_idletasks()
            lut_canvas.config(scrollregion=lut_canvas.bbox("all"))
        else:
            self.lut_window.deiconify()  # Pokaż okno tablicy LUT

class MainApp:
    def __init__(self, root):
        self.root = root
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        menubar = Menu(root)
        root.config(menu=menubar)

        # Dodawanie opcji do paska menu
        lab1_menu = Menu(menubar, tearoff=0)
        lab1_menu.add_command(label="Wczytaj obraz", command=self.load_image)
        menubar.add_cascade(label="Lab 1", menu=lab1_menu)

    def load_image(self):
        file_path = filedialog.askopenfilename(title='Select Image')
        if file_path:
            image_window = ImageWindow(file_path)



    def on_closing(self):
        self.root.quit()

def main():
    root = Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
