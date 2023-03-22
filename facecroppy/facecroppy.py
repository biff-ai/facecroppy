import cv2
from mtcnn import MTCNN

class NoFaceException(Exception): pass
class SmallImageException(Exception): pass

class FaceCroppy:
    def __init__(self, image_path, desired_size=(768, 768), margin_ratio=0.2):
        self.image = cv2.imread(image_path)
        self.height, self.width, _ = self.image.shape
        self.desired_size = desired_size
        self.margin_ratio = margin_ratio
        self.detector = MTCNN()

    def detect_faces(self):
        faces = self.detector.detect_faces(self.image)
        return faces

    def select_best_face(self, faces):
        if faces:
            face = max(faces, key=lambda x: x['confidence'])
            return face
        else:
            print("No face detected in the image.")
            return None

    def get_face_bounding_box(self, face):
        x, y, w, h = face['box']
        if w < self.desired_size[0]:
            x = max(0, int(x - (self.desired_size[0]-w)/2))
            w = self.desired_size[0]
        if h < self.desired_size[1]:
            y = max(0, int(y - (self.desired_size[1]-h)/2))
            h = self.desired_size[1]
        return x, y, w, h

    def add_margin_to_face_box(self, x, y, w, h):
        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2, y2 = min(self.width, x + w + margin_x), min(self.height, y + h + margin_y)
        return x1, y1, x2, y2

    def crop_image_with_margin(self, x1, y1, x2, y2):
        cropped_image_with_margin = self.image[y1:y2, x1:x2]
        return cropped_image_with_margin

    def add_padding(self, x1, y1, x2, y2):
        padding_x = max(0, int((self.desired_size[0] - (x2 - x1)) / 2))
        padding_y = max(0, int((self.desired_size[1] - (y2 - y1)) / 2))
        x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
        x2, y2 = min(self.width, x2 + padding_x), min(self.height, y2 + padding_y)
        return x1, y1, x2, y2

    def calculate_aspect_ratio(self, x1, y1, x2, y2):
        aspect_ratio = float(x2 - x1) / float(y2 - y1)
        return aspect_ratio

    def resize_image(self, cropped_image_with_margin, aspect_ratio):
        new_width = self.desired_size[0]
        new_height = int(new_width / aspect_ratio)

        if new_height > self.desired_size[1]:
            new_height = self.desired_size[1]
            new_width = int(new_height * aspect_ratio)

        resized_cropped_image = cv2.resize(cropped_image_with_margin, (new_width, new_height))
        return resized_cropped_image

    def add_border(self, resized_cropped_image):
        padding_top = max(0, (self.desired_size[1] - resized_cropped_image.shape[0]) // 2)
        padding_bottom = max(0, self.desired_size[1] - resized_cropped_image.shape[0] - padding_top)
        padding_left = max(0, (self.desired_size[0] - resized_cropped_image.shape[1]) // 2)
        padding_right = max(0, self.desired_size[0] - resized_cropped_image.shape[1] - padding_left)

        padded_image = cv2.copyMakeBorder(resized_cropped_image, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_image

    def resize_and_pad_image(self, cropped_image):
        height, width, _ = cropped_image.shape
        aspect_ratio = float(width) / float(height)

        new_width = self.desired_size[0]
        new_height = int(new_width / aspect_ratio)

        if new_height > self.desired_size[1]:
            new_height = self.desired_size[1]
            new_width = int(new_height * aspect_ratio)

        resized_image = cv2.resize(cropped_image, (new_width, new_height))

        padding_top = (self.desired_size[1] - new_height) // 2
        padding_bottom = self.desired_size[1] - new_height - padding_top
        padding_left = (self.desired_size[0] - new_width) // 2
        padding_right = self.desired_size[0] - new_width - padding_left

        padded_image = cv2.copyMakeBorder(resized_image, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_image


    def crop(self, output_path):
        faces = self.detect_faces()
        face = self.select_best_face(faces)

        if face:
            x, y, w, h = self.get_face_bounding_box(face)
            x1, y1, x2, y2 = self.add_margin_to_face_box(x, y, w, h)

            cropped_image_with_margin = self.image[y1:y2, x1:x2]
            padded_image = self.resize_and_pad_image(cropped_image_with_margin)

            cv2.imwrite(output_path, padded_image)
            print(f"Face cropped and saved as {output_path}.")
            return padded_image
        else:
            raise NoFaceException(f"No face detected")
            

