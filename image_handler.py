import cv2

class ImageHandler:
    def __init__(self, image_path=None):
        self.image = None
        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path):
        """Wczytuje obraz z podanej ścieżki."""
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Nie udało się wczytać obrazu z {image_path}")

    def duplicate(self):
        """Duplikuje obraz i zwraca jego kopię."""
        if self.image is None:
            raise ValueError("Obraz nie został wczytany.")
        return self.image.copy()

    def save_image(self, save_path):
        """Zapisuje obraz w podanej ścieżce."""
        if self.image is None:
            raise ValueError("Obraz nie został wczytany.")
        cv2.imwrite(save_path, self.image)