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

    # Adding Alpha Channels too
    # y1, y2 = y_offset, y_offset + sec_image.shape[0]
    # x1, x2 = x_offset, x_offset + sec_image.shape[1]
    #
    # alpha_s = sec_image[:, :, 3] / 255.0
    # alpha_l = 1.0 - alpha_s
    #
    # for c in range(0, 3):
    #     sec_image[y1:y2, x1:x2, c] = (alpha_s * sec_image[:, :, c] +
    #                               alpha_l * origin[y1:y2, x1:x2, c])
    #
    # origin[y_offset:(y_offset + sec_image.shape[0]), x_offset:(x_offset + sec_image.shape[1])] = sec_image


def put4ChannelImageOn4ChannelImage(back, fore, x, y):
    rows, cols, channels = fore.shape
    trans_indices = fore[..., 3] != 0 # Where not transparent
    overlay_copy = back[y:y+rows, x:x+cols]
    print(trans_indices.shape)
    overlay_copy[trans_indices] = fore[trans_indices]
    back[y:y+rows, x:x+cols] = overlay_copy


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


def contour_detect2(imgray):
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


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

    # # show the original image and the edge detected image
    # print("STEP 1: Edge Detection")
    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# # find the contours in the edged image, keeping only the
# # largest ones, and initialize the screen contour
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
#
# # loop over the contours
# for c in cnts:
#     # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#     # if our approximated contour has four points, then we
#     # can assume that we have found our screen
#     if len(approx) == 4:
#         screenCnt = approx
#         break
#
# # show the contour (outline) of the piece of paper
# print("STEP 2: Find contours of paper")
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # apply the four point transform to obtain a top-down
# # view of the original image
# warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
#
# # convert the warped image to grayscale, then threshold it
# # to give it that 'black and white' paper effect
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset=10, method="gaussian")
# warped = (warped > T).astype("uint8") * 255
#
# # show the original and scanned images
# print("STEP 3: Apply perspective transform")
# cv2.imshow("Original", imutils.resize(orig, height=650))
# cv2.imshow("Scanned", imutils.resize(warped, height=650))
# cv2.waitKey(0)


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = imutils.resize(frame, height=500)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = edge_detect(frame)

    #find contours
    screenCnt = contour_detect(edged)


    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
    edged = imutils.resize(edged, height=250)

    if(screenCnt is not None):
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)

    add_second_image(frame, edged)
    # put4ChannelImageOn4ChannelImage(frame, edged, 20, 20)
    # overlay_image_alpha(frame, edged, (10, 10), edged[:, :, 3] / 255.0)

    # Display the resulting frame
    cv2.imshow('origin', frame)
    # cv2.imshow('edged', edged)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()