import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
Pseudocode:

imrreading() --> The imgreading() function is responsible for opening and loading the image file, and it takes the file path 
of the image as input and reads the image using the imread function. In return, if the image cannot be read, it raises an error.

detect_circle() --> The detect_circle() function detects the largest circle in a grayscale image. As a result, it returns the center's (x,y) coordinates and the 
radius (r) of the best found circle.

main() --> The main function is the runner of the program. If a good circle is found, as a result, it then it makes a circle using the cv.circle() function, one in the center and a 
larger circle for the edges.
"""

#Function definitions / pseudocode resides above.

"""
imgreading():
    Parameters:
    path (str): Path to the image file.

    Returns:
    np.ndarray: Loaded image in BGR format.
"""
def imgreading(path):
    img = cv.imread("sodaCircle.png")
    if img is None:
        raise FileNotFoundError(f"Image not found")
    return img
"""
Parameters:
    gray (np.ndarray): Grayscale version of the input image. 
    width (int): Width of the original image (used to scale radius bounds).
Returns:
    tuple[int, int, int]: (x, y, r) center coordinates and radius of the detected circle. This uses the circle equation (x-h)^2 + (y-k)^2 = r^2
"""
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
    cv.circle(canister, (x, y), r, (0, 255, 0), 5) #parentheses is the rgb color for both lines
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



