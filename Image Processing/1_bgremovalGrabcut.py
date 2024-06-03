# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define the imshow function for displaying images
def imshow(title="Image", image=None, size= 9):
    # Calculate the aspect ratio to preserve image proportions
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    
    # Create a figure with a specified size
    plt.figure(figsize=(size * aspect_ratio, size))
    
    # Display the image using matplotlib (convert it to RGB format first)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Set the title of the plot
    plt.title(title)
    
    # Display the plot
    plt.show()

# Load and resize the input image
img = cv2.imread('images/lambo.jpg')
img = cv2.resize(img, (800, 1199))

# Display the original image
imshow("Original Image", img)

# Create an empty mask with the same size as the image
mask = np.zeros(img.shape[:2], np.uint8)

# Initialize the background and foreground models used in GrabCut algorithm
bgModel = np.zeros((1, 65), np.float64) * 255
fgModel = np.zeros((1, 65), np.float64) * 255

# Define a rectangle (ROI) that encloses the foreground object in the image
rect = (140, 480, 500, 300)

# Apply the GrabCut algorithm to extract the foreground object based on the rectangle provided
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

# Create a binary mask where pixels labeled as probable background or definite background are set to 0, and the rest to 1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the mask to the image to remove the background
img = img * mask2[:, :, np.newaxis]

# Display the image with the background removed
imshow("Background Removed", img)
