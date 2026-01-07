import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def read_image(path: str):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found")
    return img


def detect_circle(gray: np.ndarray, width: int):
    circles = cv.HoughCircles(
        cv.GaussianBlur(gray, (9, 9), 2),cv.HOUGH_GRADIENT,dp=1.2,minDist=400,param1=100,param2=50,minRadius=int(width * 0.42),maxRadius=int(width * 0.52))

    rounded_circles = np.uint16(np.around(circles))
    circle_candidates = rounded_circles[0]
    def retrieval(circle):
        return circle[2]
    largest_circle = max(circle_candidates, key=retrieval)

    x, y, r = largest_circle
    return int(x), int(y), int(r)

def main():
    canister = read_image("soda1.jpg")

    shape = canister.shape
    h = shape[0]
    w = shape[1]
    print("image width:", w)
    print("image height:", h)
    gray = cv.cvtColor(canister, cv.COLOR_BGR2GRAY)
    result = detect_circle(gray, w)
    if result is None:
        print("no circles found")
        return

    x, y, r = result
    cv.circle(canister, (x, y), r, (0, 255, 0), 5) #parentheses is color for both lines
    cv.circle(canister, (x, y), 5, (0, 0, 255), 9)

# this is the plt (matplotlib section) of my code, other is CV upwards
    rgb = cv.cvtColor(canister, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)
    plt.title("Soda can detection")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()


