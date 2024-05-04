# import cv2
# import numpy as np

debug = True

# #Display image
def display(img, frameName="OpenCV Image"):
    if not debug:
        return
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)

#rotate the image with given theta value
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
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(0,0,0))
    return rotated


# def slope(x1, y1, x2, y2):
#     if x1 == x2:
#         return 0
#     slope = (y2-y1)/(x2-x1)
#     theta = np.rad2deg(np.arctan(slope))
#     return theta

def round_to_angle_multiples(number):

    # Define the possible angles
    angles = [45, -45, 90, -90, 135, -135]

    # Calculate the absolute difference between the number and each angle
    differences = [abs(number - angle) for angle in angles]

    # Find the index of the angle with the smallest difference
    min_index = differences.index(min(differences))

    # Return the corresponding angle
    return angles[min_index]





# def main(filePath):
#     img = cv2.imread(filePath)
#     textImg = img.copy()

#     small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)

#     #find the gradient map
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

#     display(grad)

#     #Binarize the gradient image
#     _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     display(bw)

#     #connect horizontally oriented regions
#     #kernal value (9,1) can be changed to improved the text detection
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
#     connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
#     display(connected)

#     # using RETR_EXTERNAL instead of RETR_CCOMP
#     # _ , contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #opencv >= 4.0


#     mask = np.zeros(bw.shape, dtype=np.uint8)
#     #display(mask)
#     #cumulative theta value
#     cummTheta = 0
#     #number of detected text regions
#     ct = 0
#     for idx in range(len(contours)):
#         x, y, w, h = cv2.boundingRect(contours[idx])
#         mask[y:y+h, x:x+w] = 0
#         #fill the contour
#         cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
#         #display(mask)
#         #ratio of non-zero pixels in the filled region
#         r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

#         #assume at least 45% of the area is filled if it contains text
#         if r > 0.45 and w > 8 and h > 8:
#             #cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)

#             rect = cv2.minAreaRect(contours[idx])
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
#             cv2.drawContours(textImg,[box],0,(0,0,255),2)

#             #we can filter theta as outlier based on other theta values
#             #this will help in excluding the rare text region with different orientation from ususla value 
#             theta = slope(box[1][0], box[1][1], box[2][0], box[2][1])
#             cummTheta += theta
#             ct +=1 
#             print("Theta", theta)
            
#     #find the average of all cumulative theta value
#     orientation = cummTheta/ct
#     orientation=round_to_angle_multiples(orientation)
#     print(orientation)
#     print("Image orientation in degress: ", orientation)
#     finalImage = rotate(img, -orientation)
#     display(textImg, "Detectd Text minimum bounding box")
#     display(finalImage, "Deskewed Image")

# if __name__ == "__main__":
#     filePath = '1.jpeg'
#     main(filePath)
import cv2
import numpy as np
def rotate_img(img):
    # Apply Hough Lines Transform
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

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
            print(x1,y1,x2,y2)
            if x2 - x1 != 0:  # Avoid division by zero
                 orientation = np.arctan((y2 - y1) / (x2 - x1))*180  # Radians
            else:
                orientation = 180/2  # Vertical line (slope is undefined)
        
    # print(orientation)
    # Draw only the longest line (if any)
    # if longest_line is not None:
    #     x1, y1, x2, y2 = longest_line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw blue line with thickness 2
    finalImage=img
    # print(orientation)
    if (orientation>20.0 or orientation<-20):
        orientation=round_to_angle_multiples(orientation)
        finalImage = rotate(img, 180-orientation)

    return finalImage
    # print(orientation)

def preprocess(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Salt and pepper noise
    img=cv2.medianBlur(img,3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('Detected Lines (Longest)', img)
    # Apply Canny edge detection
    edges = cv2.Canny(img, 50, 150)  # Adjust threshold values as needed

    rotated_img=rotate_img(edges)

    #Binarize the gradient image
    _, img = cv2.threshold(rotated_img, 50.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # display(bw)
    
    #--morphological operations--
    #find the gradient map
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # display(img)
    return img
    


    # Display the image with the longest line (or nothing if no lines)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  

# # Example usage
image_path = 'fonts-dataset\IBM Plex Sans Arabic\887.jpeg'
preprocess(image_path)
