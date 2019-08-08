import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_lines(img):
    lines = cv2.HoughLines(img,1,np.pi/180,200)
    print(lines)
    #for rho,theta in lines[0]:
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    return img


def gradient_img(colorsrc):
    '''
        http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    '''
    SCALE = 1
    DELTA = 0
    DDEPTH = cv2.CV_16S  ## to avoid overflow

    graysrc = cv2.cvtColor(colorsrc, cv2.COLOR_BGR2GRAY)
    graysrc = cv2.GaussianBlur(graysrc, (3, 3), 0)

    ## gradient X ##
    gradx = cv2.Sobel(graysrc, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
    gradx = cv2.convertScaleAbs(gradx)
    gradx = add_lines(gradx)
    #print(gradx.shape)

    cv2.imshow("gradx", gradx)

    ## gradient Y ##
    grady = cv2.Sobel(graysrc, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)
    grady = cv2.convertScaleAbs(grady)
    cv2.imshow("grady", grady)

    grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

    return grad 

SCALE = 1
DELTA = 0
DDEPTH = cv2.CV_16S  ## to avoid overflow

img = cv2.imread('./images/stickyNote.png')
#img = cv2.imread('./images/hahaha.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.GaussianBlur(img, (11, 11), 0)
grad_img = gradient_img(img)
print(img.shape)
cv2.imshow("org", img)
cv2.imshow("gradient_img", grad_img)
cv2.waitKey(0)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#gray = cv2.GaussianBlur(gray, (3, 3), 0)
#edges = cv2.Canny(gray, 100, 200)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
gradx = cv2.Sobel(img, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
#gradx = cv2.convertScaleAbs(gradx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

#gradx = cv2.cvtColor(gradx, cv2.COLOR_BGR2GRAY)

sobelxe = np.uint8(sobelx) 
sobelye = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)


#plt.subplot(2,3,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,2),plt.imshow(cv2.cvtColor(sobelx, cv2.COLOR_BGR2RGB))
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,3),plt.imshow(gradx)
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,4),plt.imshow(sobely)
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

#plt.show()

lines = cv2.HoughLines(sobelxe,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(sobelxe,(x1,y1),(x2,y2),(0,255,0),2)

#plt.subplot(2,3,6),plt.imshow(sobelxe)
#plt.title('Sobel EDGE'), plt.xticks([]), plt.yticks([])

cv2.imshow('newImg',sobelxe)

#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
