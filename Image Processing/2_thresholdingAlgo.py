# Import necessary libraries
import cv2
import numpy as np
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

# Load the input image in BGR format
img = cv2.imread('images/lion.webp')

# Thresholding with a fixed threshold
# Convert the image to binary using a fixed threshold value of 12
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

#  Grayscale the image
# Convert the original image to grayscale
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding on grayscale image
# Convert the grayscale image to binary using a fixed threshold value of 12
retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding with Gaussian
# Apply adaptive thresholding on the grayscale image using Gaussian method
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

# Otsu's Thresholding
# Apply Otsu's thresholding on the grayscale image to automatically find the optimal threshold value
retval2, otsu = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original image and the thresholded versions
imshow('original', img)       # Original image in color (BGR format)
imshow('threshold', threshold)           # Thresholded image (fixed threshold) on color image
imshow('threshold2', threshold2)         # Thresholded image (fixed threshold) on grayscale image
imshow('gaus', gaus)                     # Adaptive thresholding using Gaussian on grayscale image
imshow('otsu', otsu)                     # Otsu's thresholding on grayscale image
