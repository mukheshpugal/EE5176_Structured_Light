import numpy as np
import cv2

imgL = (255 * np.load('codem.npy')).astype(np.uint8)
imgR = (255 * np.load('pat.npy')).astype(np.uint8)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=13)
d = stereo.compute(imgL, imgR)

# d = np.clip(d, d[50, 100], d[370, 530])
# big = [370, 530]
# small = [50, 100]

d -= np.min(d)
d = (255 * d.astype(np.float32) / np.max(d)).astype(np.uint8)

cv2.imshow('x', d)
cv2.waitKey(0)

# d = np.load('overlay.npy')
# d -= np.min(d)
# d /= np.max(d)
# d = (d * 255).astype(np.uint8)

# cv2.imwrite('./res/overlay.png', d)

