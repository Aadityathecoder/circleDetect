import numpy as np
import matplotlib.pyplot as plt
import cv2

canister = cv2.imread("soda_bottom.jpg")

gryscl = cv2.cvtColor(canister, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gryscl, 5)

circles = cv2.HoughCircles(
    gryscl,
    cv2.HOUGH_GRADIENT,
    dp=1.1,
    minDist=30,
    param1=120,
    param2=30,
    minRadius=40,
    maxRadius=90
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0]:
        cv2.circle(canister, (x, y), r, (0, 255, 0), 2)
        cv2.circle(canister, (x, y), 4, (0, 0, 255), -1)

frame = cv2.cvtColor(canister, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(frame)
plt.title("Soda can detection")
plt.axis("off")
plt.show()
