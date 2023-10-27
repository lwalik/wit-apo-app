from tkinter import filedialog, Tk, Label, Menu
from classes.image_window import ImageWindow

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
