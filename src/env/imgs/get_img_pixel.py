import cv2
import numpy as np
import pprint

# img1 = cv2.imread('wall.png', 1)
# pprint.pprint(img1.tolist())
# cv2.imshow('img', img1)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img2', img1)

n = 30
for i in range(10):
    img = np.zeros([n, n, 3], np.uint8)
    cv2.putText(img, str(i), (0, n), cv2.FONT_HERSHEY_PLAIN, n // 10 - 1, (255, 255, 255))
    # cv2.imshow('img' + str(i), img)

    # 获取二值矩阵
    img = np.where(img == 0, img, 1)
    tmp = img[:, :, 0]
    tmp2 = np.pad(tmp, ((1, 1), (1, 1)), mode='constant', constant_values=0)

cv2.waitKey(0)
