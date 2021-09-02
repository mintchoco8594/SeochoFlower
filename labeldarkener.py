#labelaug
#단순히 다른이름으로 저장하는 기능입니다.
#어두운 사진 파일명은 d_파일명 으로 만들어집니다.

import shutil
import os

path = ""  #라벨 폴더 주소
for i in os.listdir(path):
  source = "/"+str(i)
  destination = "/d_"+str(i)
  shutil.copyfile(path + source,path + destination)
