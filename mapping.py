import numpy as np
from sklearn.preprocessing import normalize
import cv2
import os

# load image from the file
imageL = cv2.imread('E:/DisparityMapping/left_frame.jpg')
imageR = cv2.imread('E:/DisparityMapping/right_frame.jpg')

# SGBM Parameters settings
window_size = 5
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160,
    blockSize=5,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

print('computing disparity...')
displ = left_matcher.compute(imageL, imageR)
dispr = right_matcher.compute(imageR, imageL)
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imageL, None, dispr)
cv2.imshow('16bit',filteredImg)
print("Before ")
print(filteredImg)
print("after")
path = 'E:/DisparityMapping/Image_Capture'
cv2.imwrite(os.path.join(path, "disparity16bit.jpg"), filteredImg)
print(filteredImg.shape)
print(filteredImg[240][320])

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)
cv2.imshow('Left_Frame', imageL)
cv2.imshow('Right_Frame', imageR)
cv2.imshow('Disparity Map', filteredImg)
print(filteredImg)
path = 'E:/DisparityMapping/Image_Capture'
cv2.imwrite(os.path.join(path, "disparity.jpg"), filteredImg)

cv2.waitKey()
cv2.destroyAllWindows()

