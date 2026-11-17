import numpy as np
import cv2

#이미지 특징 추출
'''
2d -> 1d 질감,색상,형태로
'''

img = np.zeros((100,100,3), dtype=np.uint8)
img[:,:] = (100,255,30)

hist_b = cv2.calcHist([img], [0], None, [4], [0,256])
hist_g = cv2.calcHist([img], [1], None, [4], [0,256])
hist_r = cv2.calcHist([img], [2], None, [4], [0,256])
'''
histSize = 256 -> 8
32 이렇게 해서 (100,255,30) 찾아냄
256 안해도 (100,255,30) 잘나옴
'''

# print(f"{hist_b}\n{hist_g}\n{hist_r}")

result = np.concatenate([hist_b,hist_g,hist_r])
print(result.shape)
print(result)

