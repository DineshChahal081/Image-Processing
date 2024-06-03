# Import necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Define our imshow function
def imshow(title="Image", image=None, size=10):
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

# Chessboard corner detection
img = cv2.imread('images/chessboard.png')

# Resize the image to display it in a larger size for better visualization
img = cv2.resize(img, (0, 0), fx=4, fy=4)

# Display the original image
imshow("Original Image", img)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform corner detection using the "Good Features to Track" algorithm
# Parameters: (image, maxCorners, qualityLevel, minDistance)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

# Convert the floating-point coordinates of corners to integers
corners = np.intp(corners)

# Draw circles at the detected corners
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 5, (255, 0, 1), -1)

# Connect the detected corners with lines
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])

        # Generate a random color for the line
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))

        # Draw a line between two corners with the random color
        cv2.line(img, corner1, corner2, color, 1)

# Display the image with connected chessboard corners
imshow("Connected Chessboard Corners", img)
