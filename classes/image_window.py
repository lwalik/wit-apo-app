import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import filedialog, Label, Toplevel, Menu, Frame, Canvas, Scrollbar, ttk, Scale, Button
from functions.custom_functions import calculate_histogram, check_if_monochrome, calculate_lut_arrays


class ImageWindow:
    def __init__(self, root, image_path, is_copy=False):
        self.top = Toplevel(root)
        self.top.title(os.path.basename(image_path) + (" - Kopia" if is_copy else ""))
        self.image = cv2.imread(image_path)
        self.is_monochrome = check_if_monochrome(self.image)  # Sprawdź, czy obraz jest monochromatyczny
        self.label = None  # Dodajemy atrybut label
        self.display_image()
        self.lut_window = None  # Okno tablicy LUT
        self.lut_arrays = None  # Oblicz tablice LUT

        menubar = Menu(self.top)
        self.top.config(menu=menubar)

        # Dodawanie opcji do paska menu
        lab1_menu = Menu(menubar, tearoff=0)
        lab1_menu.add_command(label="Histogram", command=self.show_histogram)
        lab1_menu.add_command(label="Tablica LUT", command=self.show_lut_tables)
        lab1_menu.add_command(label="Duplikuj", command=lambda: ImageWindow(root, image_path, is_copy=True))
        lab1_menu.add_command(label="Zapisz", command=self.save_image)
        lab2_menu = Menu(menubar, tearoff=0)
        lab2_menu.add_command(label="Rozciąganie liniowe", command=self.linear_stretching)
        lab2_menu.add_command(label="Rozciąganie nieliniowe", command=self.gamma_stretching)
        lab2_menu.add_command(label="Wyrównanie histogramu", command=self.histogram_equalization)

        menubar.add_cascade(label="Lab 1", menu=lab1_menu)
        menubar.add_cascade(label="Lab 2", menu=lab2_menu)

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
            hist = calculate_histogram(channel)
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

    def show_lut_tables(self):
        self.lut_arrays = calculate_lut_arrays(self.image, self.is_monochrome)
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

    def linear_stretching(self):
        def apply_linear_stretching():
            p1 = min_scale.get()
            p2 = max_scale.get()
            q3 = new_min_scale.get()
            q4 = new_max_scale.get()
            self.image = perform_linear_stretching(self.image, p1, p2, q3, q4)
            self.display_image()

        def perform_linear_stretching(image, p1, p2, q3, q4):
            # Tworzenie kopii obrazu, aby nie modyfikować oryginalnego obrazu
            result = np.copy(image)
            # Tworzenie maski logicznej dla pikseli, które znajdują się w zakresie [p1, p2]
            # Piksele poza tym zakresem nie będą modyfikowane
            mask = (image >= p1) & (image <= p2)
            # Stosowanie rozciągania liniowego dla pikseli w zakresie [p1, p2]
            # Nowa wartość piksela = ((stara wartość piksela - p1) * ((q4 - q3) / (p2 - p1)) + q3)
            # Wynik jest rzutowany na typ uint8, aby mieścił się w zakresie wartości pikseli obrazu
            result[mask] = ((image[mask] - p1) * ((q4 - q3) / (p2 - p1)) + q3).astype(np.uint8)
            # Ustawienie wartości pikseli poniżej p1 na q3 oraz p2 na q4
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

    def gamma_stretching(self):
        def apply_gamma_stretching():
            gamma = gamma_scale.get()
            saturation = saturation_scale.get()
            self.image = perform_gamma_stretching(self.image, gamma, saturation)
            self.display_image()

        def perform_gamma_stretching(image, gamma, saturation):
            # Normalizacja obrazu do zakresu [0, 1]
            image_normalized = image / 255.0
            # Obliczanie procentu pikseli do przesycenia
            pixels_to_saturate = int(image.size * saturation / 100)
            # Sortowanie wartości pikseli
            sorted_pixel_values = np.sort(image_normalized, axis=None)
            # Znalezienie wartości progowych dla przesycenia
            low_staturation_threshold = sorted_pixel_values[pixels_to_saturate]
            high_staturation_threshold = sorted_pixel_values[-pixels_to_saturate - 1]
            # Obcięcie wartości pikseli
            image_clipped = np.clip(image_normalized, low_staturation_threshold, high_staturation_threshold)
            # Normalizacja obrazu do zakresu [0, 1] po odcięciu
            image_clipped = (image_clipped - low_staturation_threshold) / (high_staturation_threshold - low_staturation_threshold)
            # Obliczenie transformacji gamma
            image_gamma = np.power(image_clipped, gamma)
            # # Skalowanie obrazu z powrotem do zakresu [0, 255]
            image_gamma = np.clip(image_gamma * 255, 0, 255).astype(np.uint8)
            return image_gamma

        gamma_window = Toplevel(self.top)
        gamma_window.title("Rozciąganie Gamma")
        gamma_window.geometry('250x300')

        gamma_scale = Scale(gamma_window, from_=0.1, to=3.0, resolution=0.1, orient="horizontal", label="Gamma",
                            width=10, length=200)
        gamma_scale.pack()
        gamma_scale.set(1.0)

        saturation_scale = Scale(gamma_window, from_=0, to=5, orient="horizontal", label="Przesycenie (%)",
                                 width=10, length=200)
        saturation_scale.pack()
        saturation_scale.set(0)

        apply_button = Button(gamma_window, text="Zastosuj", command=apply_gamma_stretching)
        apply_button.pack()

    def histogram_equalization(self):

        def equalize_histogram(image, histogram):
            cdf = histogram.cumsum()
            cdf_min = cdf.min()
            cdf_max = cdf.max()
            image_equalized = ((cdf[image] - cdf_min) / (cdf_max - cdf_min) * 255).astype(np.uint8)
            return image_equalized

        if self.is_monochrome:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            histogram = calculate_histogram(gray_image)
            equalized_image = equalize_histogram(gray_image, histogram)
            self.image = cv2.merge([equalized_image] * 3)
        else:
            b, g, r = cv2.split(self.image)
            b_histogram = calculate_histogram(b)
            g_histogram = calculate_histogram(g)
            r_histogram = calculate_histogram(r)
            b_equalized = equalize_histogram(b, b_histogram)
            g_equalized = equalize_histogram(g, g_histogram)
            r_equalized = equalize_histogram(r, r_histogram)
            self.image = cv2.merge([b_equalized, g_equalized, r_equalized])

        self.display_image()

