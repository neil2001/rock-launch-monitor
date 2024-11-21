import cv2
import numpy as np

# Load your two images
image1 = cv2.imread("output/frames/frame_0002.jpg")
image2 = cv2.imread("output/frames/frame_0003.jpg")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply thresholding for high contrast
_, thresh1 = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)

# Find contours in the images
contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour corresponds to the tape
contour1 = max(contours1, key=cv2.contourArea)
contour2 = max(contours2, key=cv2.contourArea)

# Approximate the contours to get a simpler representation
epsilon1 = 0.01 * cv2.arcLength(contour1, True)
epsilon2 = 0.01 * cv2.arcLength(contour2, True)
approx1 = cv2.approxPolyDP(contour1, epsilon1, True)
approx2 = cv2.approxPolyDP(contour2, epsilon2, True)

# Draw contours on the original images
image1_with_contours = image1.copy()
image2_with_contours = image2.copy()

# Get the moments of the contours
moments1 = cv2.moments(approx1)
moments2 = cv2.moments(approx2)

# Calculate the orientation (angle of rotation)
angle1 = 0.5 * np.arctan2(2 * moments1['mu11'], moments1['mu20'] - moments1['mu02']) * (180 / np.pi)
angle2 = 0.5 * np.arctan2(2 * moments2['mu11'], moments2['mu20'] - moments2['mu02']) * (180 / np.pi)


# Angular displacement
angular_displacement = angle2 - angle1

# If you know the time between frames, compute the spin rate
time_between_frames = 1.0/80.0  # seconds (adjust based on your data)
angular_velocity_deg_per_sec = angular_displacement / time_between_frames  # degrees per second

# Convert angular velocity to RPM
rpm = (angular_velocity_deg_per_sec / 360) * 60

print(f"Angular Displacement: {angular_displacement:.2f} degrees")
print(f"Spin Rate: {rpm:.2f} RPM")
# Draw contours and orientation on images
cv2.drawContours(image1, [approx1], -1, (0, 255, 0), 2)
cv2.drawContours(image2, [approx2], -1, (0, 255, 0), 2)

combined = np.hstack((image1, image2))
cv2.imshow("Processed Images", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()