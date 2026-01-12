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

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't recieve frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
#--------------------

def imgreading(path):
    img = cv.imread("sodaCircle.png")
    return img
"""
Parameters:
    gray (np.ndarray): Grayscale version of the input image. 
    width (int): Width of the original image.
Returns:
    tuple[int, int, int]: (x, y, r) center coordinates and radius of the detected circle. This uses the circle equation (x-h)^2 + (y-k)^2 = r^2
"""
def detect_circle(gray: np.ndarray, width: int):
    circles = cv.HoughCircles(
        cv.GaussianBlur(gray, (9, 9), 2),cv.HOUGH_GRADIENT,dp=1.2,minDist=400,param1=100,param2=50,minRadius=int(width * 0.42),maxRadius=int(width * 0.52))

    rounded_circles = np.uint16(np.around(circles))
    circle_possible = rounded_circles[0]
    def retrieval(circle):
        return circle[2]
    largest_circle = max(circle_possible, key=retrieval)

    x, y, r = largest_circle
    return int(x), int(y), int(r)

def main():
    canister = imgreading("soda1.jpg")

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
    cv.circle(canister, (x, y), r, (0, 255, 0), 5) #Parentheses is the rgb color for both lines, and for example it is the rgb (0,255,0) inside the circle creation
    cv.circle(canister, (x, y), 5, (0, 0, 255), 9)

# This is the plt (matplotlib section) of my code, other is CV upwards.
    rgb = cv.cvtColor(canister, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("Soda can detection")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()



