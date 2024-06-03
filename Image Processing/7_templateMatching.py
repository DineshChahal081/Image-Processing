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

# Load the original image and the template image
img = cv2.imread('images/soccer_practice.jpg', 0)   # Load the original image in grayscale
template = cv2.imread('images/ball.PNG', 0)        # Load the template image in grayscale
h, w = template.shape                              # Get the height and width of the template

# Define the methods used for template matching
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# Display the original image
imshow("Original Image", img)

# Display the template image
imshow("Template", template)

# Iterate over each method and perform template matching
for method in methods:
    # Make a copy of the original image to draw the result
    img2 = img.copy()

    # Perform template matching for the current method
    result = cv2.matchTemplate(img2, template, method)

    # Find the minimum and maximum values and their respective locations in the result
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Choose the appropriate location based on the method used (some methods require minimum location)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    # Calculate the bottom-right point of the rectangle based on the template size and location
    bottom_right = (location[0] + w, location[1] + h)

    # Draw a rectangle around the matched area in the image
    cv2.rectangle(img2, location, bottom_right, 255, 5)

    # Display the result for the current method
    imshow(f"Match (Method: {method})", img2)
