#이미지데이터 증강툴입니다.


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"



aug = ImageDataGenerator(rotation_range =30,  # 좌우로 30도(degree)범위 내에서 랜덤회전
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         shear_range=0.2,
                         brightness_range=(0.2,1.0), #밝기조절 정도
                         horizontal_flip= True,  
                         vertical_flip = True,   
                         fill_mode='nearest') # 마지막 옵션 주의하자.



# carnation_pink DONE                                      김진성(완)
# carnation_red DONE                                       김진성
# chrysanthemum_baegseon DONE                              서정연(완)
# chrysanthemum_disbud DONE                                서정연
# gentian_gentian DONE
# =============================
# gerbera_gerbera DONE                                     권보경(완)
# chrysanthemum_cosmosDONE
# daisy_daisy DONE                                         최윤수
# gypsophila_overtimeDONE
# hydrangea_greenDONE                                      권보경
# =============================
# hydrangea_pink
# lily_medusa
# lily-siberiaDONE                                         조은호
# lisianthus_jollypink수집중
# lisianthus_lavenderDONE
# ==============================
# rose_fuegoDONE
# rose_hazelDONE
# sunflower_sunflowerDONE        http://naver.me/x0avCWIP
# sunflower_teddybear DONE       http://naver.me/5EQodvs3
# daisy_double

path = 'C:/Users/jinsung/Downloads/newflower/daisy_double'    #이미지경로 지정

augoutput= 'C:/Users/jinsung/Downloads/flowersaug/daisy_double' #저장할경로 지정


# with tf.device('/GPU:0'):  #gpu사용여부
for i in range(1, 80):
    if os.path.exists(path+"/cosmos"+str(i)+".png"): #품종명으로 수정해줘야 합니다
      image = plt.imread(path+"/cosmos"+str(i)+".png") #품종명으로 수정해줘야 합니다
      image = np.expand_dims(image, axis=0)
      imageGen = aug.flow(image,
                      batch_size=1,
                      save_to_dir=augoutput,
                      save_prefix="chrysanthemum_cosmos"+str(i),
                      save_format='png')
      total = 0
      for image in imageGen:
          # 루프가 돌면서 이미지가 한장씩 생성된다.
          total += 1
          # N장 채우면 멈추자, 숫자수정시 그만큼 찍어냅니다
          if total == 14:
              break
      total = 0
    else:
      pass
