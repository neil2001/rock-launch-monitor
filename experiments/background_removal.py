import cv2
import os
from rembg import remove

video_path = "data/IMG_0260.mov"
output_folder = "output/frames"

FRAME_RATE = 3

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    if frame_count % FRAME_RATE == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
        frame_no_background = remove(frame)
        cv2.imwrite(frame_filename, frame_no_background)
        saved_frame_count += 1
        print(saved_frame_count)

    frame_count += 1

cap.release()
print(f"Extracted {saved_frame_count} frames to '{output_folder}'.")
