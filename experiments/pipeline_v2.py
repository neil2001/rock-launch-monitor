import cv2
import numpy as np

# Load your two images
image1 = cv2.imread("output/frames/frame_0002.jpg")
image2 = cv2.imread("output/frames/frame_0003.jpg")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Use Shi-Tomasi Corner Detection to find corners (good features to track)
corners1 = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners1 = np.int32(corners1)

# Create a mask image for drawing purposes
mask = np.zeros_like(image1)

# Use optical flow to track corners from image1 to image2
corners2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, np.float32(corners1), None)

# Draw the corners on the images for visualization
for i in range(len(corners1)):
    x1, y1 = corners1[i].ravel()
    x2, y2 = corners2[i].ravel()
    x2, y2 = int(x2), int(y2)

    print(x1,y1,x2,y2)

    # Draw initial corner (in green)
    image1 = cv2.circle(image1, (x1, y1), 5, (0, 255, 0), -1)

    # Draw tracked corner (in red)
    image2 = cv2.circle(image2, (x2, y2), 5, (0, 0, 255), -1)

    # Draw the line between initial and tracked position (in blue)
    image2 = cv2.line(image2, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Combine images side by side to visualize the movement of corners
combined = np.hstack((image1, image2))

# Display the images
cv2.imshow("Tracked Corners - Before and After", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()