import cv2
import matplotlib.pyplot as plt

# Load the image (grayscale is preferred for SIFT)
image_path = 'output/frames/frame_0004.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Unable to load image.")
    exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display the image with keypoints
plt.figure(figsize=(10, 6))
plt.imshow(image_with_keypoints, cmap='gray')
plt.title(f"SIFT Keypoints ({len(keypoints)} found)")
plt.axis('off')
plt.show()

# Optional: Print descriptors
print(f"Number of Keypoints: {len(keypoints)}")
print(f"Descriptor Shape: {descriptors.shape}")
