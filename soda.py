import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
Pseudocode:

read_image() --> The read_image() function is responsiblwe for opening and loading the image file, and it takes the file path 
of the image as input, reads the image using the imread function, and if the image cannot be read, it raises an error.

detect_circle() --> The detect_circle() function detects the largest circle in a grayscale image, and it uses CV'S HoughCircles
method to find circular objects in the image. It specifically looks for circles that match the width of the soda can, and once it
identifies potential circles, it picks the largest one based on radius. As a result, it returns the center's (x,y) coordinates and the
radius (r) of the best found circle.

main() --> The main function is the runner of the program. Essentially what it does is it reads the image file provided ("soda1.jpg"),
then determines the width and height of the image. Afterwards, it calls the detect_circle() function in the function which I defined above
to detect the soda can in the image. If a good circle is found, then it makes a circle using  the cv.circle() function, one in the center and a 
larger circle for the edges.
"""

def read_image(path):
    img = cv.imread("sodaCircle.png")
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



