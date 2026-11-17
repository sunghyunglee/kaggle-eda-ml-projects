import numpy as np
import cv2

#이미지 특징 추출

img = np.zeros((100,100,3), dtype= np.uint8)
img[:,:] = (100,255,30)

hist_b = cv2.calcHist([img], [0], None, [4], [0,256])
hist_g = cv2.calcHist([img], [1], None, [4], [0,256])
hist_r = cv2.calcHist([img], [2], None, [4], [0,256])

result = np.concatenate([hist_b, hist_g, hist_r])
print(result.shape)
print(result)

