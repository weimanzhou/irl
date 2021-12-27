import cv2
import numpy as np


path = './img.png'

#
# img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
# print(img.shape)
# img = np.where(img != 0, img, 255)
# cv2.imshow('IMREAD_UNCHANGED', img)
# img = cv2.imread(path, cv2.IMREAD_COLOR)
# print(img.shape)
# cv2.imshow('IMREAD_COLOR', img)
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('IMREAD_GRAYSCALE', img)
# print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
# cv2.imshow('img2', img)
# print(img)

img2 = np.full((5, 5, 3), 0, np.uint8)
cv2.imshow(
    'img1', img2
)
img2 = np.full((5, 5, 3), 255, np.uint8)
cv2.imshow(
    'img2', img2
)
# print(img2)
# cv2.imshow('img3', img2)
# img2[0:20, 0:20] = img
# cv2.imshow('img3-1', img2)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
# print(img2)
# cv2.imshow('img4', img2)

# obs = np.random.randint(0, 255, ((self.rows + 2) * 20, (self.cols + 2 + self.cols // 2 + self.cols + 2) * 20, 3),
#                         np.uint8)

cv2.waitKey(0)