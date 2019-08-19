# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def add_second_image(origin, sec_image):
    x_offset = origin.shape[1] - 20
    y_offset = origin.shape[0] - 20
    # x_offset = y_offset = 20
    # origin[y_offset:y_offset + sec_image.shape[0], x_offset:x_offset + sec_image.shape[1]] = sec_image
    origin[y_offset - sec_image.shape[0]:y_offset, x_offset - sec_image.shape[1]:x_offset] = sec_image


def contour_detect(edge):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    return screenCnt


def edge_detect(image):
    # ratio = image.shape[0] / 500.0
    # orig = image.copy()
    # image = imutils.resize(image, height=100)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    return edged


def text_area_detect(img):
    rgb = cv2.pyrDown(img)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #For opencv 3+ comment the previous line and uncomment the following line
    #_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    mask = np.zeros(bw.shape, dtype=np.uint8)
    
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
    
    cv2.imshow('rects', rgb)


"""For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain straight even after the transformation. 
To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. 
Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the function cv2.getPerspectiveTransform. 
Then apply cv2.warpPerspective with this 3x3 transformation matrix."""
def draw_contour(contour, img):
    min_square_size = 987
    images = []
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(screenCnt)
    # If the contour is not really small, or really big
    h,w = img.shape[0], img.shape[1]
    if area > min_square_size and area < h*w-(2*(h+w)):
        # Get the four corners of the contour
        epsilon = .1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        print("========================================================")

        # now that we have our screen contour, we need to determine
        # the top-left, top-right, bottom-right, and bottom-left
        # points so that we can later warp the image -- we'll start
        # by reshaping our contour to be our finals and initializing
        # our output rectangle in top-left, top-right, bottom-right,
        # and bottom-left order
        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")
 
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
 
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        
        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        
        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
        	[0, 0],
        	[maxWidth - 1, 0],
        	[maxWidth - 1, maxHeight - 1],
        	[0, maxHeight - 1]], dtype = "float32")
        
        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        cv2.imshow('warp', warp)
        cv2.waitKey()

        text_area_detect(warp)
        cv2.waitKey()

        print("========================================================")

        images.append(warp.copy())
    return images


# Capture frame-by-frame
frame = cv2.imread("./images/receipt.jpg") 
frame = imutils.resize(frame, height=500)

# Our operations on the frame come here
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = edge_detect(frame)

cv2.imshow('edged', edged)
cv2.waitKey()

# find contours
screenCnt = contour_detect(edged)

edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
edged = imutils.resize(edged, height=250)

if (screenCnt is not None):
    #cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
    images = draw_contour(screenCnt, frame)
    add_second_image(frame, images[0])

# Display the resulting frame
cv2.imshow('origin', frame)

cv2.waitKey()

# When everything done, release the capture
cv2.destroyAllWindows()
