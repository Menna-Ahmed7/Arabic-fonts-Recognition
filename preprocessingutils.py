# import cv2
# import numpy as np
from os.path import  join
from PIL import Image


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

    print(img[0][0])
    # rotate orignal image to show transformation
    print(img[0][0])

    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(int(img[0][0]),int(img[0][0]),int(img[0][0])))
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
            print(x1,y1,x2,y2)
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
    # print(orientation)

def preprocess(image_path):
    # Read the image
    originalimg = cv2.imread(image_path)

    # Salt and pepper noise
    img=cv2.medianBlur(originalimg,5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # display(img)

    # cv2.imshow('Detected Lines (Longest)', img)
    # Apply Canny edge detection
    edges = cv2.Canny(img, 20, 150)  # Adjust threshold values as needed

    rotated_img=rotate_img(edges,img)
    # display(img)

    if (img[0][0]>200):
        _, img = cv2.threshold(rotated_img, 50.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, img = cv2.threshold(rotated_img, 50.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    # binary_img=rotated_img
    # contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(originalimg, contours, -1, (0, 255, 0), 2)  # Green color, thickness 2
    # display(originalimg)
    # binary_img=img

    # img=rotated_img
    # display(rotated_img)
    # display(img)

    #Binarize the gradient image
    # display(bw)
    # display(img)
    
    #--morphological operations--
    #find the gradient map
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # display(img)
    # display(originalimg)

    return img
    


    # Display the image with the longest line (or nothing if no lines)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  

# # Example usage
# image_path = '11.jpeg'
# img=preprocess(image_path)
##########OCR
# import cv2
# import pytesseract

# # Load the image
# img = cv2.imread('22.jpeg')

# # Convert the image to grayscale
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Use pytesseract to perform OCR and get bounding box coordinates
# detections = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

# print(detections)
# # Iterate over each detection
# prev=detections['top'][0]
# for i, text in enumerate(detections['text']):
#     # Skip empty detections
#     if text.strip() == '':
#         continue
    
#     if detections['top'][i] -  
#     # Extract bounding box coordinates
#     x, y, w, h = detections['left'][i], detections['top'][i], detections['width'][i], detections['height'][i]
    
#     # Extract the line image
#     line_image = img[y:y+h, x:x+w]
    
#     # Save the line image
#     cv2.imwrite(f'line_{i+1}.jpg', line_image)


def extractImageLines(img,img_name,output_dir,img_path):
    
    #cv2.RETR_EXTERNAL retrieves only the outermost contours of connected regions.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sorting contours by their start position in y just to be processed from top to bottom
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    cur_line=[]
    all_lines_found=[]
    prev_y=-1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        #compare aspect ratio w/h and area to filter contours
        aspect_ratio = w / h
        # aspect_ratio = w / float(h)

        area=cv2.contourArea(contour)
        if aspect_ratio <8 and aspect_ratio>0.3 and area >90:
        # if 0.1 < aspect_ratio < 10 and cv2.contourArea(contour) > 100:
            # print("here")
            #if valid contour check if it belongs to previous line or it is the first contotur tot work on
            if prev_y == -1 or abs(prev_y - y) < 17:
            # if  len(all_lines_found)==0 or abs(prev_y -y) <20:
                cur_line.append(contour)
            else:
                all_lines_found.append(cur_line)
                cur_line=[]
                cur_line.append(contour)
            prev_y=y

    # For the last line
    if cur_line:
        all_lines_found.append(cur_line)
    # print(all_lines_found)
    #then we have 2d array where for each line we have array of contours
    #we want to find bounding box for each line
    index=0
    for line in all_lines_found:
        # print(index)
        #concatenates all contours into single one
        line_contours = np.concatenate(line)
        x, y, w, h = cv2.boundingRect(line_contours)
        new_img = img[y:y+h, x:x+w]
        name=img_name+'_'+str(index+1)+'.jpeg'
        image = Image.fromarray(new_img)  # Convert to PIL Image object
        new_path = join(output_dir, name)
    
        # new_path = join(output_dir, name)
        image.save(new_path)
        # cv2.imwrite(f'{name}.jpeg', new_img)
        index=index+1

# extractImageLines(img,'11')
# cv2.drawContours(img, lines, -1, (0, 255, 0), 2)  # Green color, thickness 2
