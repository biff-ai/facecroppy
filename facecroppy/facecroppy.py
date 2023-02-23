import cv2
from mtcnn import MTCNN

class NoFaceException(Exception): pass
class SmallImageException(Exception): pass

class FaceCroppy:
  def __init__(self, image_path, desired_size=(512, 768)):
    self.desired_size = desired_size
    self.image_path = image_path

  def detect_faces(self, pixels):
      # create the detector, using default weights
      detector = MTCNN()
      
      # detect faces in the image
      faces = detector.detect_faces(pixels)

      # if no faces
      if len(faces) == 0:
        raise NoFaceException(f"No face detected")

      return faces
  
  def get_facebox(self, faces):
        x1, y1, width, height = list(faces[0]['box'])
        x1, y1, width, height = int(x1), int(y1), int(width), int(height)
        # extract face from the first face
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

  def resize_and_crop(self, image, desired_size=(512, 768)):
      height, width, _ = image.shape
      ratio = min(width/desired_size[0], height/desired_size[1])
      new_width, new_height = int(width/ratio), int(height/ratio)
      image = cv2.resize(image, (new_width, new_height))
      height, width, _ = image.shape
      left = int((width - desired_size[0]) / 2)
      top = int((height - desired_size[1]) / 2)
      right = left + desired_size[0]
      bottom = top + desired_size[1]
      image = image[top:bottom, left:right]
      return image


  def load_image(self):
      # load image from file
      pixels = cv2.imread(self.image_path)

      # Check if any axis is smaller than desired size
      if self.desired_size[0] >= pixels.shape[1] or self.desired_size[1] >= pixels.shape[0]:
        raise SmallImageException(f"""TODO: Upscale? Image is too small. Axis need to be minimum {self.desired_size[0]} x {self.desired_size[1]} px while image is {pixels.shape[1]} x {pixels.shape[0]}""")

      return pixels



  def crop(self):

      # load image
      pixels = self.load_image()

      # detect faces in the image
      try:
        faces = self.detect_faces(pixels) 
        x1, y1, x2, y2, width, height =  self.get_facebox(faces)
        face_pixels = pixels[y1:y2, x1:x2]
      except NoFaceException:
        print(f"""No face detected in {self.image_path}
      
        Warning: Image will be center-cropped!
        
        """)

        return self.resize_and_crop(pixels, self.desired_size)

      # calculate the ratio between the desired size and the face size
      ratio = max(self.desired_size[0] / face_pixels.shape[1], self.desired_size[1] / face_pixels.shape[0])
      
      # expand the face area to meet the desired size if the face is smaller
      if ratio > 1:
          print(ratio)
          desired_ratio = self.desired_size[0]/self.desired_size[1]
          # calculate the new width and height of the face
          new_width = int(face_pixels.shape[1] * (ratio/ (3 if ratio > 4 else 1.5)))
          new_height = int(new_width / desired_ratio)
          
          # calculate the new x and y of the face
          x_diff = x1 - (new_width - width) // 2
          x = max(0, x_diff)
          #if x_diff < 0:
          #  new_width -= abs(x_diff)

          y_diff =  y1 - (new_height - height) // 2
          y = max(0, y_diff)
          #if y_diff < 0:
            #new_height -= abs(y_diff)

          print(x1, y1, x_diff, y_diff)
          # make sure the crop won't exceed the original image size
          new_width = min(new_width, pixels.shape[1] - x_diff)
          new_height = min(new_height, pixels.shape[0] - y_diff)
          
          # crop the original image with the face in the center
          face_pixels = pixels[y:y+new_height - (y_diff if y_diff < 0 else 0), x:x+new_width - (x_diff if x_diff < 0 else 0)]

          # resize the face to the desired size
          face_pixels = cv2.resize(face_pixels, self.desired_size)
      
          return face_pixels

      else:
          padding = 1
          w_add = int(width * padding)
          h_add = int(height * padding)
          x1 -= w_add
          y1 -= h_add
          width += 2 * w_add
          height += 2 * h_add

          # Make sure the cropping area stays within the image bounds
          if x1 < 0:
              x1 = 0
          if y1 < 0:
              y1 = 0
          if x1 + width > pixels.shape[1]:
              width = pixels.shape[1] - x1
          if y1 + height > pixels.shape[0]:
              height = pixels.shape[0] - y1

          # Crop the face
          face = pixels[y1:y1+height, x1:x1+width]

          # Resize the face to desired size
          face = cv2.resize(face, self.desired_size)

          return face


