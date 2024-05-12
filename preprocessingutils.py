
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from os import listdir, mkdir
from os.path import isfile, join, dirname, exists
from PIL import Image
import os
# from sklearn.svm import SVC
from scipy.signal import convolve2d
# from sklearn.externals import joblib
# import joblib
import matplotlib.pyplot as pyplot

# rotate the image with given theta value
def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)
    
    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(int(img[0][0]),int(img[0][0]),int(img[0][0])))
    return rotated
def round_to_angle_multiples(number):

    # Define the possible angles
    angles = [45, -45, 90, -90, 135, -135]

    # Calculate the absolute difference between the number and each angle
    differences = [abs(number - angle) for angle in angles]

    # Find the index of the angle with the smallest difference
    min_index = differences.index(min(differences))

    # Return the corresponding angle
    return angles[min_index]


def rotate_img(edgeimage,originalimage):
    # Apply Hough Lines Transform
    lines = cv2.HoughLinesP(edgeimage, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    finalImage=originalimage
    # Find the longest line
    longest_line = None
    max_length = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line length using distance formula
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_length:
                max_length = length
                longest_line = line

        if longest_line is not None:
            x1, y1, x2, y2 = longest_line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                 orientation = np.arctan((y2 - y1) / (x2 - x1))*180  # Radians
            else:
                orientation = 180/2  # Vertical line (slope is undefined)
            # print("oo",orientation)
            if (orientation>20.0 or orientation<-20):
                orientation=round_to_angle_multiples(orientation)
                finalImage = rotate(originalimage, 180-orientation)
        
        # # print(orientation)
        # # Draw only the longest line (if any)
        # if longest_line is not None:
        #     x1, y1, x2, y2 = longest_line[0]
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw blue line with thickness 2
        #     display(img,"box")

    return finalImage



def preprocess(image_path,b=False):
    # Read the image
    if not b :
        img = cv2.imread(image_path)
    else:
        img = image_path


    # Salt and pepper noise
    img=cv2.medianBlur(img,5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # display(img)

    # cv2.imshow('Detected Lines (Longest)', img)
    # Apply Canny edge detection
    edges = cv2.Canny(img, 20, 150)  # Adjust threshold values as needed

    rotated_img=rotate_img(edges,img)
    # display(img)
    if (img[0][0]>200):
        _, img = cv2.threshold(rotated_img, 50.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        print("white")
    else:
        _, img = cv2.threshold(rotated_img, 50.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print("black")
    # display(img)
    return img