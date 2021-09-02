#image darkener
#이미지 명도를 100만큼 줄여 밤에 찍은 사진처럼 변환합니다
#이미 밤에찍은 사진은 넣지말아주세요
#파일명은 d_파일명 으로 만들어집니다.
import numpy as np
import cv2
import os

path = 'C:/'  #이미지 폴더 주소

for i in os.listdir(path):
  img = cv2.imread(path+"/"+str(i),cv2.IMREAD_COLOR)
  val = 100
  array = np.full(img.shape, (val,val,val),dtype=np.uint8)
  sub = cv2.subtract(img,array)
  cv2.imwrite(path+"/d_"+str(i)+'.png', sub)
