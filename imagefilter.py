import os
import cv2

path = os.getcwd()

img = cv2.imread(path + '\data\\testing\\pl3.jpg')

filtered_img = cv2.GaussianBlur(img, (25,25), 0)

filtered_img[75:325, 75:225] = img[75:325, 75:225]
cv2.imwrite('pl3.jpg', filtered_img)