import numpy as np
import cv2
from matplotlib import pyplot as plt

# Function to perform color detection based on a given color range
def color_detection(image, lower_color, upper_color):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a binary mask for the specified color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    
    # Apply bitwise AND operation to the image and the mask to extract the color regions
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

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

# Load the image
image = cv2.imread('images/colordet.jpeg')

# Define the color range for detection (in HSV format)
lower_blue = np.array([90, 50, 50])   # Lower bound for blue in HSV
upper_blue = np.array([130, 255, 255])   # Upper bound for blue in HSV


# Define the color range for black detection (in HSV format)
#lower_black = np.array([0, 0, 0])       # Lower bound for black in HSV
#upper_black = np.array([180, 255, 30])  # Upper bound for black in HSV


# Perform color detection based on the defined color range
result = color_detection(image, lower_blue, upper_blue)

# Display the original image and the color detection result
imshow("Original Image", image)   # Display the original input image
imshow("Color Detection Result", result)   # Display the result of color detection

