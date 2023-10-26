import cv2
from tkinter import filedialog, Tk, Label, Toplevel, Menu, Frame, Canvas, Scrollbar, ttk
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
        self.lut_arrays = self.calculate_lut_arrays(self.image)  # Oblicz tablice LUT

        menubar = Menu(self.top)
        self.top.config(menu=menubar)

        # Dodawanie opcji do paska menu
        plik_menu = Menu(menubar, tearoff=0)
        plik_menu.add_command(label="Histogram", command=self.show_histogram)
        plik_menu.add_command(label="Tablica LUT", command=self.show_lut_tables)
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

        if len(image.shape) == 2 or (
                len(image.shape) == 3 and np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
            image[:, :, 0], image[:, :, 2])):
            hist = self.calculate_and_plot_histogram(image, axs[0], 'gray')
            axs[0].set_title('Intensity (weighted)')

        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = self.calculate_and_plot_histogram(gray_image, axs[0], 'gray')
            axs[0].set_title('Intensity (weighted)')

        self.show_statistics(axs[0], hist, gray_image)

        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = self.calculate_and_plot_histogram(image[:, :, 2 - i], axs[i + 1], color)
            axs[i + 1].set_title(f'{color.upper()}')
            self.show_statistics(axs[i + 1], hist, image[:, :, 2 - i])

        for ax in axs:
            ax.set_xlabel("Wartość piksela")
            ax.set_ylabel("Liczba pikseli")
            ax.set_xlim([0, 256])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True, side='top')

    def calculate_and_plot_histogram(self, channel, ax, color):
        if channel is not None:
            hist = np.zeros(256)
            for value in range(256):
                hist[value] = np.sum(channel == value)
            ax.bar(range(256), hist, color=color, alpha=0.6)
            return hist
        return None

    def show_statistics(self, ax, hist, channel):
        if channel is not None:
            non_zero_hist = hist[hist > 0]
            if len(non_zero_hist) > 0:
                median = np.median(channel)
                min_val = np.min(channel)
                max_val = np.max(channel)
                mean_val = round(np.mean(channel), 3)
                std_dev = np.std(channel)
                ax.text(1.05, 0.8, f'Mediana: {median:.2f}\nMin: {min_val}\nMax: {max_val}\nŚrednia: {mean_val}\nOdch. std.: {std_dev:.2f}', transform=ax.transAxes)

    def calculate_lut_arrays(self, image):
        lut_arrays = {}
        if len(image.shape) == 2 or (
                len(image.shape) == 3 and np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
                image[:, :, 0], image[:, :, 2])):
            lut_arrays['Intensity (weighted)'] = self.calculate_lut_array(image)
            for color in ['R', 'G', 'B']:
                lut_arrays[color] = np.zeros(256, dtype=np.uint32)
        elif len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lut_arrays['Intensity (weighted)'] = self.calculate_lut_array(gray_image)
            for i, color in enumerate(['R', 'G', 'B']):
                lut_arrays[color] = self.calculate_lut_array(image[:, :, 2-i])

        return lut_arrays

    def calculate_lut_array(self, channel):
        lut_array = np.zeros(256, dtype=np.uint32)
        height, width = channel.shape[:2]
        for i in range(height):
            for j in range(width):
                pixel_value = channel[i, j]
                lut_array[pixel_value] += 1
        return lut_array

    def show_lut_tables(self):
        if self.lut_window is None or not self.lut_window.winfo_exists():
            self.lut_window = self.create_lut_window(self.lut_arrays)
        else:
            self.lut_window.deiconify()

    def create_lut_window(self, lut_arrays):
        lut_window = Toplevel()
        lut_window.title(f"Tablica LUT - {self.top.title()}")
        lut_window.geometry("955x270")

        for name in ["Intensity (weighted)", "R", "G", "B"]:
            lut_array = lut_arrays[name]
            lut_frame = Frame(lut_window)
            lut_frame.grid(row=0, column=["Intensity (weighted)", "R", "G", "B"].index(name), padx=10, pady=10)

            Label(lut_frame, text=name + (" (Pusta)" if np.sum(lut_array) == 0 else "")).pack()

            lut_canvas = Canvas(lut_frame)
            lut_canvas.pack(fill='both', expand=True)

            scrollbar = Scrollbar(lut_canvas, orient="vertical")
            scrollbar.pack(side="right", fill="y")

            lut_tree = ttk.Treeview(lut_canvas, columns=("Poziom jasności", "Liczba wystąpień"), show="headings")
            lut_tree.heading("Poziom jasności", text="Poziom jasności")
            lut_tree.heading("Liczba wystąpień", text="Liczba wystąpień")

            lut_tree.column("Poziom jasności", width=100, anchor="center")
            lut_tree.column("Liczba wystąpień", width=100, anchor="center")

            lut_tree.pack(fill='both', expand=True)

            lut_tree.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=lut_tree.yview)

            for j in range(256):
                lut_tree.insert("", "end", values=(j, lut_array[j]))

        return lut_window

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.image)


class MainApp:
    def __init__(self, root):
        self.root = root
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.title("Fajna Apka")
        root.geometry('300x70')

        header_label = Label(root, text='Projekt Wykonał')
        header_label.config(font=('Courier', 18))
        header_label.pack()
        name_label = Label(root, text='Łukasz Walicki')
        name_label.config(font=('Courier', 18))
        name_label.pack()

        menubar = Menu(root)
        root.config(menu=menubar)

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
