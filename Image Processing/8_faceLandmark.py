# Import necessary libraries
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt

# Define the imshow function for displaying images
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
# download shape_predictor_68_face_landmarks.dat file
# Define the path to the shape predictor file
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Initialize the shape predictor and face detector
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# Custom exceptions for face detection errors
class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    # Detect and extract facial landmarks from the image
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    # Draw circles and annotations on the image for each landmark point
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

# Load the input image
image = cv2.imread('images/grace.jpg')

# Display the original image
imshow('Input Image', image)

# Get the facial landmarks for the image
landmarks = get_landmarks(image)

# Annotate the image with the facial landmarks
image_with_landmarks = annotate_landmarks(image, landmarks)

# Display the result
imshow('Result', image_with_landmarks)
