import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import filedialog, Label, Toplevel, Menu, Frame, Canvas, Scrollbar, ttk, Scale, Button


class ImageWindow:
    def __init__(self, root, image_path, is_copy=False):
        self.top = Toplevel(root)
        self.top.title(os.path.basename(image_path) + (" - Kopia" if is_copy else ""))
        self.image = cv2.imread(image_path)
        self.is_monochrome = self.check_if_monochrome(self.image)  # Sprawdź, czy obraz jest monochromatyczny
        self.label = None  # Dodajemy atrybut label
        self.display_image()
        self.lut_window = None  # Okno tablicy LUT
        self.lut_arrays = None  # Oblicz tablice LUT

        menubar = Menu(self.top)
        self.top.config(menu=menubar)

        # Dodawanie opcji do paska menu
        plik_menu = Menu(menubar, tearoff=0)
        plik_menu.add_command(label="Histogram", command=self.show_histogram)
        plik_menu.add_command(label="Tablica LUT", command=self.show_lut_tables)
        plik_menu.add_command(label="Duplikuj", command=lambda: ImageWindow(root, image_path, is_copy=True))
        plik_menu.add_command(label="Zapisz", command=self.save_image)
        plik_menu.add_command(label="Rozciąganie liniowe", command=self.linear_stretching)

        menubar.add_cascade(label="Plik", menu=plik_menu)

    def display_image(self):
        b, g, r = cv2.split(self.image)
        img = cv2.merge((r, g, b))
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        if self.label is None:
            self.label = Label(self.top, image=imgtk)
            self.label.pack()
        else:
            self.label.configure(image=imgtk)
        self.label.imgtk = imgtk  # Keep a reference to the image object to prevent garbage collection

    def show_histogram(self):
        histogram_window = Toplevel()
        histogram_window.title("Histogram - " + self.top.title())
        num_histograms = 1 if self.is_monochrome else 5
        height = 3 if self.is_monochrome else 10
        fig, axs = plt.subplots(num_histograms, 1, figsize=(6, height))  # Utwórz odpowiednią liczbę subplots

        image = self.image.copy()

        if self.is_monochrome:
            hist = self.calculate_and_plot_histogram(image, axs, 'gray')
            axs.set_title('Intensity')
            self.show_statistics(axs, hist, image)
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = self.calculate_and_plot_histogram(gray_image, axs[0], 'gray')
            axs[0].set_title('Intensity (weighted)')
            self.show_statistics(axs[0], hist, gray_image)

            b, g, r = cv2.split(image)
            unweighted_image = np.round((r.astype(np.uint32) + g.astype(np.uint32) + b.astype(np.uint32)) / 3).astype(
                np.uint8)
            hist = self.calculate_and_plot_histogram(unweighted_image, axs[1], 'gray')
            axs[1].set_title('Intensity (unweighted)')
            self.show_statistics(axs[1], hist, unweighted_image)

            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = self.calculate_and_plot_histogram(image[:, :, 2 - i], axs[i + 2], color)
                axs[i + 2].set_title(f'{color.upper()}')
                self.show_statistics(axs[i + 2], hist, image[:, :, 2 - i])

        for ax in (axs if isinstance(axs, np.ndarray) else [axs]):
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
                mean_val = round(np.mean(channel), 2)
                std_dev = np.std(channel)
                ax.text(1.05, 0.2,
                        f'Mediana: {median:.2f}\nMin: {min_val}\nMax: {max_val}\nŚrednia: {mean_val}\nOdch. std.: {std_dev:.2f}',
                        transform=ax.transAxes)

    def calculate_lut_arrays(self, image):
        lut_arrays = {}
        if self.is_monochrome:
            lut_arrays['Intensity'] = self.calculate_lut_array(image)
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lut_arrays['Intensity (weighted)'] = self.calculate_lut_array(gray_image)
            b, g, r = cv2.split(image)
            unweighted_image = np.round((r.astype(np.uint32) + g.astype(np.uint32) + b.astype(np.uint32)) / 3).astype(
                np.uint8)
            lut_arrays['Intensity (unweighted)'] = self.calculate_lut_array(unweighted_image)
            for i, color in enumerate(['R', 'G', 'B']):
                lut_arrays[color] = self.calculate_lut_array(image[:, :, 2 - i])

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
        self.lut_arrays = self.calculate_lut_arrays(self.image)
        self.lut_window = self.create_lut_window(self.lut_arrays)

    def create_lut_window(self, lut_arrays):
        lut_window = Toplevel()
        lut_window.title(f"Tablica LUT - {self.top.title()}")
        width = 235 if self.is_monochrome else 1185
        lut_window.geometry(f'{width}x270')

        if self.is_monochrome:
            names = ["Intensity"]
        else:
            names = ["Intensity (weighted)", "Intensity (unweighted)", "R", "G", "B"]

        for name in names:
            lut_array = lut_arrays[name]
            lut_frame = Frame(lut_window)
            lut_frame.grid(row=0, column=names.index(name), padx=10, pady=10)

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

    def check_if_monochrome(self, image):
        if len(image.shape) == 2:
            return True
        if len(image.shape) == 3 and np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
                image[:, :, 0], image[:, :, 2]):
            return True
        return False

    def linear_stretching(self):
        def apply_linear_stretching():
            p1 = min_scale.get()
            p2 = max_scale.get()
            q3 = new_min_scale.get()
            q4 = new_max_scale.get()
            self.image = perform_linear_stretching(self.image, p1, p2, q3, q4)
            self.display_image()

        def perform_linear_stretching(image, p1, p2, q3, q4):
            result = np.copy(image)
            mask = (image >= p1) & (image <= p2)
            result[mask] = ((image[mask] - p1) * ((q4 - q3) / (p2 - p1)) + q3).astype(np.uint8)
            result[image < p1] = q3
            result[image > p2] = q4
            return result

        stretching_window = Toplevel(self.top)
        stretching_window.title("Rozciąganie liniowe")
        stretching_window.geometry('250x300')

        min_scale = Scale(stretching_window, from_=0, to=255, orient="horizontal", label="Minimum", width=10, length=200)
        min_scale.pack()
        min_scale.set(0)

        max_scale = Scale(stretching_window, from_=0, to=255, orient="horizontal", label="Maksimum", width=10, length=200)
        max_scale.pack()
        max_scale.set(255)

        new_min_scale = Scale(stretching_window, from_=0, to=255, orient="horizontal", label="Nowe Minimum", width=10, length=200)
        new_min_scale.pack()
        new_min_scale.set(0)

        new_max_scale = Scale(stretching_window, from_=0, to=255, orient="horizontal", label="Nowe Maksimum", width=10, length=200)
        new_max_scale.pack()
        new_max_scale.set(255)

        apply_button = Button(stretching_window, text="Zastosuj", command=apply_linear_stretching)
        apply_button.pack()
