import cv2
import numpy as np
import matplotlib.pyplot as plt

canister = cv2.imread("IMG_0182.jpg.jpg")

if canister is None:
    print("Error: Could not load image")
    exit()

gryscl = cv2.cvtColor(canister, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gryscl, 5)

# Detect circles
circles = cv2.HoughCircles(
    img,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=50,
    param2=30,
    minRadius=30,
    maxRadius=400
)

if circles is not None:
    circles = np.uint16(np.around(circles))

    # Select ONLY the largest circle (by radius)
    largest_circle = max(circles[0], key=lambda c: c[2])
    x, y, r = largest_circle

    # Draw only this one circle
    cv2.circle(canister, (x, y), r, (0, 255, 0), 3)
    cv2.circle(canister, (x, y), 4, (0, 0, 255), -1)

    print(f"Detected circle at ({x}, {y}) with radius {r}")
else:
    print("No circles detected. Try adjusting parameters.")

frame = cv2.cvtColor(canister, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(frame)
plt.title("Soda can detection")
plt.axis("off")
plt.show()
