import cv2
import numpy as np

# Loading image contains lines
img = cv2.imread('./images/sudoku.jpg')
# Convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Apply Canny edge detection, return will be a binary image
edges = cv2.Canny(gray,50,100,apertureSize = 3)
# Apply Hough Line Transform, minimum lenght of line is 200 pixels
lines = cv2.HoughLines(edges,1,np.pi/180,200)
# Print and draw line on the original image
for rho,theta in lines[0]:
 print(rho, theta)
 a = np.cos(theta)
 b = np.sin(theta)
 x0 = a*rho
 y0 = b*rho
 x1 = int(x0 + 1000*(-b))
 y1 = int(y0 + 1000*(a))
 x2 = int(x0 - 1000*(-b))
 y2 = int(y0 - 1000*(a))
 cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Show the result
cv2.imshow("Line Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
