import os
import cv2
import numpy as np

path = "C:/Users/jinsung/Downloads/flowersaug/output"

for i in os.listdir(path):
  img = cv2.imread(path+"/"+str(i),cv2.IMREAD_COLOR)
  val = 100
  array = np.full(img.shape, (val,val,val),dtype=np.uint8)
  sub = cv2.subtract(img,array)
  cv2.imwrite(path+"/d_"+str(i)+'.png', sub)
