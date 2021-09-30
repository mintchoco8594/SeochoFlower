import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"

path = "C:/Users/jinsung/Downloads/flowersaug/carnation_pink"

augoutput= "C:/Users/jinsung/Downloads/flowersaug/output"

aug = ImageDataGenerator(rotation_range =30,  # 좌우로 30도(degree)범위 내에서 랜덤회전
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         shear_range=0.2,
                         brightness_range=(0.2,1.0),
                         horizontal_flip=True,
                         vertical_flip = True,
                         fill_mode='nearest')


with tf.device('/GPU:0'):
  for i in range(1, 80):
    if os.path.exists(path+"/carnation_pink"+str(i)+".png"):
      image = plt.imread(path+"/carnation_pink"+str(i)+".png")
      image = np.expand_dims(image, axis=0)
      imageGen = aug.flow(image,
                      batch_size=1,
                      save_to_dir=augoutput,
                      save_prefix="carnation_pink"+str(i),
                      save_format='png')
      total = 0
      for image in imageGen:
          # 루프가 돌면서 이미지가 한장씩 생성된다.
          total += 1
          # 15장 채우면 멈추자
          if total == 15:
              break
      total = 0
    else:

      pass
