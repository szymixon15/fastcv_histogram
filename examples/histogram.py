import cv2
import torch
import fastcv

img = cv2.imread("artifacts/test.jpg", 1) #BGR
img_tensor = torch.from_numpy(img).cuda()

cv2_hist = cv2.calcHist([img], [1], None, [256], [0, 256])

print("HISTOGRAM CV2")
print(cv2_hist)

histogram_tensorcub = fastcv.histogram_cub(img_tensor)
histogram_tensorthrust = fastcv.histogram_thrust(img_tensor)

print("HISTOGRAM CUB")
print(histogram_tensorcub)

print("HISTOGRAM THRUST")
print(histogram_tensorthrust)

