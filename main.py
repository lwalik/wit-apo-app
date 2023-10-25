import cv2
from tkinter import filedialog, Tk, Button, Label, Toplevel, Menu, BOTH, Frame, Canvas, Scrollbar
from PIL import Image, ImageTk
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ImageWindow:
    def __init__(self, root, image_path, is_copy=False):
        self.top = Toplevel(root)
        self.top.title(os.path.basename(image_path) + (" - Kopia" if is_copy else ""))
        self.image = cv2.imread(image_path)
        self.display_image()
        self.lut_window = None  # Okno tablicy LUT
        self.lut_array = self.calculate_lut_array(self.image)  # Oblicz tablicę LUT

        menubar = Menu(self.top)
        self.top.config(menu=menubar)

        # Dodawanie opcji do paska menu
        plik_menu = Menu(menubar, tearoff=0)
        plik_menu.add_command(label="Histogram", command=self.show_histogram)
        plik_menu.add_command(label="Tablica LUT", command=self.show_lut_table)
        plik_menu.add_command(label="Duplikuj", command=lambda: ImageWindow(root, image_path, is_copy=True))
        plik_menu.add_command(label="Zapisz", command=self.save_image)

        menubar.add_cascade(label="Plik", menu=plik_menu)

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
        fig, axs = plt.subplots(4, 1, figsize=(6, 8))  # Utwórz 4 subplots

        image = self.image.copy()

        # Histogram dla obrazów monochromatycznych
        if len(image.shape) == 2 or (len(image.shape) == 3 and np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 0], image[:, :, 2])):
            self.calculate_and_plot_histogram(image, axs[0], 'gray')
            axs[0].set_title('Monochromatyczny')
            for i in range(1, 4):
                axs[i].set_title(f'Kanał {["R", "G", "B"][i-1]} (brak)')
        else:
            axs[0].set_title('Monochromatyczny (brak)')
            # Histogramy dla kanałów RGB
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                self.calculate_and_plot_histogram(image[:, :, i], axs[i + 1], color)
                axs[i + 1].set_title(f'Kanał {color.upper()}')

        for ax in axs:
            ax.set_xlabel("Wartość piksela")
            ax.set_ylabel("Liczba pikseli")
            ax.set_xlim([0, 256])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=BOTH, expand=True, side='top')

    def calculate_and_plot_histogram(self, channel, ax, color):
        if channel is not None:
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

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.image)

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
            ImageWindow(self.root, file_path)

    def on_closing(self):
        self.root.quit()

def main():
    root = Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
