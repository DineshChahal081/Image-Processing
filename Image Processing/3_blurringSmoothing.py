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

# Read the image from file
image_path = 'images/blurr_smooth.jpg'  # Replace with the actual image file path
img = cv2.imread(image_path)

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Define the lower and upper bounds for the blue color in HSV format
lower_blue = np.array([90, 50, 50])     # Lower bound for blue in HSV
upper_blue = np.array([130, 255, 255])  # Upper bound for blue in HSV

# Create a mask for blue color within the defined range
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply the mask to the original image to extract blue regions
res = cv2.bitwise_and(img, img, mask=mask)

#  Create a kernel for smoothing operations
kernel = np.ones((15, 15), np.float32) / 225

# Apply bitwise_and to the original image with the mask to get smoothed image
smoothed = cv2.bitwise_and(img, img, mask=mask)

# Apply Gaussian blur to the masked region
blur = cv2.GaussianBlur(res, (15, 15), 0)

# Apply median blur to the masked region
median = cv2.medianBlur(res, 15)

#Display the original image and intermediate results
imshow('Original Image', img)  # Display the original image
imshow('Mask', mask)           # Display the binary mask for the blue color
imshow('Res', res)             # Display the result of extracting blue regions
imshow('Smoothed', smoothed)   # Display the smoothed version of the image
imshow('Blur', blur)           # Display the result of Gaussian blur on the blue regions
imshow('Median', median)       # Display the result of median blur on the blue regions
