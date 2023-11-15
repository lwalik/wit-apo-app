import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import filedialog, Label, Toplevel, Menu, Frame, Canvas, Scrollbar, ttk, Scale, Button, simpledialog, messagebox
from functions.custom_functions import calculate_histogram, check_if_monochrome, calculate_lut_arrays, update_scale


class ImageWindow:
    def __init__(self, root, image_path, is_copy=False, image=None):
        self.top = Toplevel(root)
        self.top.title(os.path.basename(image_path) + (" - Kopia" if is_copy else ""))
        self.image = image if is_copy else cv2.imread(image_path)
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
        lab1_menu.add_command(label="Duplikuj", command=lambda: ImageWindow(root, image_path, is_copy=True, image=self.image))
        lab1_menu.add_command(label="Zapisz", command=self.save_image)

        lab2_menu = Menu(menubar, tearoff=0)
        lab2_menu.add_cascade(label="Typ", menu=self.create_type_submenu(lab2_menu))
        lab2_menu.add_command(label="Rozciąganie liniowe", command=self.linear_histogram_stretching)
        lab2_menu.add_command(label="Rozciąganie nieliniowe", command=self.gamma_stretching)
        lab2_menu.add_command(label="Wyrównanie histogramu", command=self.histogram_equalization)
        lab2_menu.add_command(label="Negacja", command=self.negation)
        lab2_menu.add_command(label="Redukcja poziomów szarości", command=self.gray_level_reduction)
        lab2_menu.add_command(label="Progowanie binarne", command=self.binary_thresholding)
        lab2_menu.add_command(label="Progowanie z zachowaniem poziomów szarości", command=self.gray_level_thresholding)
        lab2_menu.add_command(label="Rozciąganie liniowe z danymi użytkownika", command=self.linear_stretching)

        lab3_menu = Menu(menubar, tearoff=0)
        lab3_menu.add_command(label="Dodaj obrazy z wysyceniem", command=self.add_images_with_saturation)
        lab3_menu.add_command(label="Dodaj obrazy bez wysycenia", command=self.add_images_without_saturation)
        lab3_menu.add_command(label="Dodaj obraz przez liczbę", command=self.add_image_with_number)
        lab3_menu.add_command(label="Pomnóż obraz przez liczbę", command=self.multiply_image_with_number)
        lab3_menu.add_command(label="Podziel obraz przez liczbę", command=self.divide_image_with_number)
        lab3_menu.add_command(label="Różnica bezwzględna obrazów", command=self.absolute_difference)
        lab3_menu.add_command(label="NOT", command=self.perform_not_operation)
        lab3_menu.add_command(label="AND", command=self.perform_and_operation)
        lab3_menu.add_command(label="OR", command=self.perform_or_operation)
        lab3_menu.add_command(label="XOR", command=self.perform_xor_operation)

        menubar.add_cascade(label="Lab 1", menu=lab1_menu)
        menubar.add_cascade(label="Lab 2", menu=lab2_menu)
        menubar.add_cascade(label="Lab 3", menu=lab3_menu)

    def create_type_submenu(self, parent_menu):
        type_submenu = Menu(parent_menu, tearoff=0)
        type_submenu.add_command(label="8-bit", command=lambda: self.convert_to_grayscale())
        type_submenu.add_command(label="Binary", command=lambda: self.convert_to_binary())
        return type_submenu


    def display_image(self):
        if len(self.image.shape) == 2:
            img = Image.fromarray(self.image)
        else:
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
        stretching_window.geometry('300x300')

        min_scale_frame = Frame(stretching_window)
        min_scale_frame.pack(pady=5)
        Button(min_scale_frame, text="<", command=lambda: update_scale(min_scale, -1)).pack(side="left")
        min_scale = Scale(min_scale_frame, from_=0, to=255, orient="horizontal", label="Minimum", width=10, length=200)
        min_scale.pack(side="left", padx=5)
        min_scale.set(0)
        Button(min_scale_frame, text=">", command=lambda: update_scale(min_scale, 1)).pack(side="left")

        max_scale_frame = Frame(stretching_window)
        max_scale_frame.pack(pady=5)
        Button(max_scale_frame, text="<", command=lambda: update_scale(max_scale, -1)).pack(side="left")
        max_scale = Scale(max_scale_frame, from_=0, to=255, orient="horizontal", label="Maksimum", width=10, length=200)
        max_scale.pack(side="left", padx=5)
        max_scale.set(255)
        Button(max_scale_frame, text=">", command=lambda: update_scale(max_scale, 1)).pack(side="left")

        new_min_scale_frame = Frame(stretching_window)
        new_min_scale_frame.pack(pady=5)
        Button(new_min_scale_frame, text="<", command=lambda: update_scale(new_min_scale, -1)).pack(side="left")
        new_min_scale = Scale(new_min_scale_frame, from_=0, to=255, orient="horizontal", label="Nowe Minimum", width=10,
                              length=200)
        new_min_scale.pack(side="left", padx=5)
        new_min_scale.set(0)
        Button(new_min_scale_frame, text=">", command=lambda: update_scale(new_min_scale, 1)).pack(side="left")

        new_max_scale_frame = Frame(stretching_window)
        new_max_scale_frame.pack(pady=5)
        Button(new_max_scale_frame, text="<", command=lambda: update_scale(new_max_scale, -1)).pack(side="left")
        new_max_scale = Scale(new_max_scale_frame, from_=0, to=255, orient="horizontal", label="Nowe Maksimum",
                              width=10, length=200)
        new_max_scale.pack(side="left", padx=5)
        new_max_scale.set(255)
        Button(new_max_scale_frame, text=">", command=lambda: update_scale(new_max_scale, 1)).pack(side="left")

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
        gamma_window.geometry('300x180')

        gamma_frame = Frame(gamma_window)
        gamma_frame.pack(pady=5)
        Button(gamma_frame, text="<", command=lambda: update_scale(gamma_scale, -0.1)).pack(side="left")
        gamma_scale = Scale(gamma_frame, from_=0.1, to=3.0, resolution=0.1, orient="horizontal", label="Gamma",
                            width=10, length=200)
        gamma_scale.pack(side="left", padx=5)
        gamma_scale.set(1.0)
        Button(gamma_frame, text=">", command=lambda: update_scale(gamma_scale, 0.1)).pack(side="left")

        saturation_frame = Frame(gamma_window)
        saturation_frame.pack(pady=5)
        Button(saturation_frame, text="<", command=lambda: update_scale(saturation_scale, -1)).pack(side="left")
        saturation_scale = Scale(saturation_frame, from_=0, to=5, orient="horizontal", label="Przesycenie (%)",
                                 width=10, length=200)
        saturation_scale.pack(side="left", padx=5)
        saturation_scale.set(0)
        Button(saturation_frame, text=">", command=lambda: update_scale(saturation_scale, 1)).pack(side="left")

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

    def convert_to_grayscale(self):
        # Sprawdź, czy obraz jest już w formacie 8-bitowym (odcienie szarości)
        if len(self.image.shape) == 2 or (len(self.image.shape) == 3 and self.image.shape[2] == 1):
            messagebox.showinfo("Informacja", "Obraz jest już w formacie 8-bitowym.")
            return

        # Konwersja obrazu na 8-bitowy format odcieni szarości
        converted_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.image = converted_image
        self.is_monochrome = True
        self.display_image()

    def convert_to_binary(self):
        # Sprawdź, czy obraz jest już w formacie binarnym
        if np.array_equal(self.image, self.image.astype(bool).astype(np.uint8) * 255):
            messagebox.showinfo("Informacja", "Obraz jest już w formacie binarnym.")
            return

        # Użyj progowania, aby przekształcić obraz na format binarny
        _, binary_image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)

        self.image = binary_image
        self.is_monochrome = True
        self.display_image()

    def negation(self):
        if not self.is_monochrome:
            self.convert_to_grayscale()
        self.image = 255 - self.image
        self.display_image()

    def gray_level_reduction(self):
        levels = simpledialog.askinteger("Redukcja poziomów szarości", "Podaj liczbę poziomów szarości (1-256):",  minvalue=1, maxvalue=256)
        if levels:
            self.image = (self.image // (256 // levels) * (256 // levels)).astype(np.uint8)
            self.display_image()

    def binary_thresholding(self):
        threshold = simpledialog.askinteger("Progowanie binarne", "Podaj próg (0-255):", minvalue=0, maxvalue=255)
        if threshold is not None:
            _, self.image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
            self.display_image()

    def gray_level_thresholding(self):
        threshold = simpledialog.askinteger("Progowanie z zachowaniem poziomów szarości", "Podaj próg (0-255):",
                                            minvalue=0, maxvalue=255)
        if threshold is not None:
            self.image[self.image < threshold] = 0
            self.display_image()

    def linear_histogram_stretching(self):
        if not self.is_monochrome:
            self.convert_to_grayscale()
        min_val = np.min(self.image)
        max_val = np.max(self.image)
        print('min_val: ', min_val, ' max_val: ', max_val)
        if min_val != max_val:
            self.image = ((self.image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        self.display_image()

    def add_images_with_saturation(self):
        file_path = filedialog.askopenfilename(title='Wybierz obraz do dodania')
        if file_path:
            image_to_add = cv2.imread(file_path)
            if image_to_add.shape != self.image.shape:
                messagebox.showerror("Błąd", "Obrazy muszą mieć ten sam rozmiar i liczbę kanałów.")
                return

            # Dodawanie obrazów z przycinaniem wartości do maksymalnego zakresu
            # Konwersja obrazów na int16, aby zapobiec przepełnieniu
            result = np.clip(self.image.astype(np.int16) + image_to_add.astype(np.int16), 0, 255).astype(np.uint8)

            # Opcjonalnie: konwersja wyniku na odcienie szarości
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            self.image = result
            self.is_monochrome = check_if_monochrome(self.image)
            self.display_image()


    def add_images_without_saturation(self):
        file_path = filedialog.askopenfilename(title='Wybierz obraz do dodania')
        if file_path:
            image_to_add = cv2.imread(file_path)
            if image_to_add.shape != self.image.shape:
                messagebox.showerror("Błąd", "Obrazy muszą mieć ten sam rozmiar i liczbę kanałów.")
                return

            # Dodawanie obrazów z przycinaniem wartości do maksymalnego zakresu
            # Konwersja obrazów na int16, aby zapobiec przepełnieniu
            self.image = np.clip(self.image.astype(np.int16) + image_to_add.astype(np.int16), 0, 255).astype(
                np.uint8)
            self.is_monochrome = check_if_monochrome(self.image)
            self.display_image()
            np.invert()

    def add_image_with_number(self):
        number = simpledialog.askfloat("Dodawanie obrazu przez liczbę", "Podaj liczbę:", minvalue=0)
        self.is_monochrome = check_if_monochrome(self.image)
        if number is not None:
            # Sprawdzenie, czy obraz jest monochromatyczny
            if self.is_monochrome:
                self.image = np.clip(self.image.astype(np.int16) + number, 0, 255).astype(np.uint8)
            else:
                # Konwersja każdego kanału koloru oddzielnie
                for i in range(3):
                    self.image[:, :, i] = np.clip(self.image[:, :, i].astype(np.int16) + number, 0, 255).astype(
                        np.uint8)

            self.display_image()

    def multiply_image_with_number(self):
        number = simpledialog.askfloat("Mnożenie obrazu przez liczbę", "Podaj liczbę:", minvalue=0)
        self.is_monochrome = check_if_monochrome(self.image)
        if number is not None:
            if self.is_monochrome:
                self.image = np.clip(self.image.astype(np.int16) * number, 0, 255).astype(np.uint8)
            else:
                # Konwersja każdego kanału koloru oddzielnie
                for i in range(3):
                    self.image[:, :, i] = np.clip(self.image[:, :, i].astype(np.int16) * number, 0, 255).astype(
                        np.uint8)

            self.display_image()

    def divide_image_with_number(self):
        number = simpledialog.askfloat("Dzielenie obrazu przez liczbę", "Podaj liczbę:", minvalue=1)
        self.is_monochrome = check_if_monochrome(self.image)
        if number is not None:
            if self.is_monochrome:
                self.image = np.clip(self.image.astype(np.int16) // number, 0, 255).astype(np.uint8)
            else:
                # Konwersja każdego kanału koloru oddzielnie
                for i in range(3):
                    self.image[:, :, i] = np.clip(self.image[:, :, i].astype(np.int16) // number, 0, 255).astype(
                        np.uint8)

            self.display_image()

    def absolute_difference(self):
        file_path = filedialog.askopenfilename(title='Wybierz obraz do różnicy bezwzględnej')
        if file_path:
            image_to_subtract = cv2.imread(file_path)
            if image_to_subtract.shape != self.image.shape:
                messagebox.showerror("Błąd", "Obrazy muszą mieć ten sam rozmiar i liczbę kanałów.")
                return
            self.image = np.abs(self.image.astype(int) - image_to_subtract.astype(int)).astype(np.uint8)
            self.display_image()

    def perform_not_operation(self):
        self.image = np.invert(self.image)
        self.display_image()

    def perform_and_operation(self):
        second_image_path = filedialog.askopenfilename(title='Wybierz drugi obraz')
        if second_image_path:
            second_image = cv2.imread(second_image_path)
            self.image = np.bitwise_and(self.image, second_image)
            self.display_image()

    def perform_or_operation(self):
        second_image_path = filedialog.askopenfilename(title='Wybierz drugi obraz')
        if second_image_path:
            second_image = cv2.imread(second_image_path)
            self.image = np.bitwise_or(self.image, second_image)
            self.display_image()

    def perform_xor_operation(self):
        second_image_path = filedialog.askopenfilename(title='Wybierz drugi obraz')
        if second_image_path:
            second_image = cv2.imread(second_image_path)
            self.image = np.bitwise_xor(self.image, second_image)
            self.display_image()