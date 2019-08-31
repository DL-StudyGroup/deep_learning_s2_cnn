# import necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import os, sys
import cv2
import imutils
from hand_writing import pred_matrix


def add_second_image(origin, sec_image):
    x_offset = origin.shape[1] - 20
    y_offset = origin.shape[0] - 20
    origin[y_offset - sec_image.shape[0]:y_offset, x_offset - sec_image.shape[1]:x_offset] = sec_image


def contour_detect(edge):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    h,w = edge.shape[0], edge.shape[1]
    min_square_size = h * w / 3  
    print('height: ' + str(h) + ' width = ' + str(w))
    print('min_square= ' + str(min_square_size))

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        area = np.int(area)
        print('contour_detect area = ' + str(area))
        print('approx = ' + str(len(approx)))

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4 and (area > min_square_size):
            screenCnt = approx
            print(area)
            #cv2.drawContours(edge, [screenCnt], -1, (255, 255, 0), 2)
            #cv2.imshow('origin', edge)
            #cv2.waitKey()
            break
    print('final result: ')
    print('contour_detect area = ' + str(area))
    print('approx = ' + str(len(approx)))
    return screenCnt


def edge_detect(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    return edged


def text_area_detect(img):
    #rgb = cv2.pyrDown(img)
    #rgb = img
    small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    locs = []

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
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        ar = w / float(h)
    
        # since credit cards used a fixed size fonts with 4 groups
        # of 4 digits, we can prune potential contours based on the
        # aspect ratio
        if ar > 2.5 and ar < 4.0:
            # contours can further be pruned on minimum/maximum width
            # and height
            print(ar)
            print('w is ' + str(w) + ' and height is ' + str(h))
            if (w > 90 and w < 100) and (h > 28 and h < 35):
                locs.append((x, y, w, h))
                #cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
                #roi = img[y:y+h, x:x+w]
                #cv2.imshow(str(idx), roi)
    
    return locs 


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
     
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
     
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
     
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
     
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def extract_text_images(gray, locs):
    # initialize the list of group digits
    rois = []
    pad = 5
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        #group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        group = cv2.threshold(group, 127, 255, cv2.THRESH_BINARY_INV)[1]
        
        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right
        digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = imutils.grab_contours(digitCnts)
        digitCnts = sort_contours(digitCnts,method="left-to-right")[0]
        # loop over the digit contours
        for cnt, c in enumerate(digitCnts):
            print(cv2.contourArea(c))
            # compute the bounding box of the individual digit, extract
            # the digit, and resize it to have the same fixed size as
            # the reference OCR-A images
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y - pad :y + h + pad, x - pad:x + w + pad]
            #cv2.imshow(str(i) + '-' + str(cnt) + '-----.jpg', roi)
            roi = cv2.resize(roi.copy(), (28, 28))
            cv2.imshow(str(i) + '-' + str(cnt) + '-----.jpg', roi)
            rois.append(roi)
                
    return rois

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
    print('area == ' + str(area))
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

        #cv2.imshow('warp', warp)
        #cv2.waitKey()
        locs = text_area_detect(warp)

        print("========================================================")

        images.append(warp.copy())
    return images, locs


# Capture frame-by-frame
#frame = cv2.imread("./images/receipt.jpg") 
# frame = cv2.imread("./images/credit_1.png")
# frame = cv2.imread("./images/credit_2.png")
frame = cv2.imread("./images/vcard.png")
#frame = cv2.imread("./images/credit.jpg")
frame = imutils.resize(frame, height=500)

# convert the image to grayscale, blur it, and find edges
# in the image
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#gray = cv2.bilateralFilter(gray, 11, 17, 17)
#edged = cv2.Canny(gray, 0, 200)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)


cv2.imshow('edged', edged)
cv2.waitKey()

# find contours
screenCnt = contour_detect(edged)


if (screenCnt is not None):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>There is contour...')
    images, locs = draw_contour(screenCnt, frame)
    img_arr = []
    if(len(images) > 0):
        add_second_image(frame, images[0])
        gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        if(len(locs) > 0):
            rois = extract_text_images(gray, locs)
            if(len(rois) > 0):
                for i, roi in enumerate(rois):
                    h,w = roi.shape[0], roi.shape[1]
                    # print('height: ' + str(h) + ' width = ' + str(w))
                    img_arr.append(pred_matrix(roi , "../hand_writing/param/cnn_params.pkl")[0])
                    cv2.imshow(str(i) + '.jpg' , roi)
            cv2.waitKey()
            print("img_arr", img_arr)
else:
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>There is no contour...')

# Display the resulting frame
cv2.imshow('origin', frame)

cv2.waitKey()

# When everything done, release the capture
cv2.destroyAllWindows()
