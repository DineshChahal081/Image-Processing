# Import necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Define the imshow function for displaying images
def imshow(title="Image", image=None, size=10):
    # Create a figure with a specified size
    plt.figure(figsize=(size, size))
    
    # Display the image using matplotlib (convert it to RGB format first)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Set the title of the plot
    plt.title(title)
    
    # Turn off axis display
    plt.axis("off")
    
    # Display the plot
    plt.show()

# Load the image from file
image_path = 'images/murphy.jpeg'  # Replace with the actual image file path
img = cv2.imread(image_path)

# Display the input image
imshow('Input Image', img)

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Convert the image to grayscale for better processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image using the face cascade classifier
# Parameters: (image, scaleFactor, minNeighbors, minSize)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over detected faces
for (x, y, w, h) in faces:
    # Draw a rectangle around the detected face on the original image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    # Region of Interest (ROI) for the detected face in grayscale and color
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    
    # Detect eyes in the ROI of the face using the eye cascade classifier
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    
    # Iterate over detected eyes within the face ROI
    for (ex, ey, ew, eh) in eyes:
        # Draw a rectangle around the detected eye on the color image
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

# Display the original image with face and eye detections
imshow('Image with Face and Eye Detection', img)
