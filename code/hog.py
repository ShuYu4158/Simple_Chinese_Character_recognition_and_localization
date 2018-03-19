import numpy as np
import os
#from cyvlfeat.hog import hog
from skimage.io import imread
import glob
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog


def chinese_filter(num):
  #Reading image
  filename = '/home/henry/new_hog/data/pei/pei%d.PNG'%num
  img = imread(filename)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  resize = cv2.resize(gray, (100,100))
  feature = hog(resize, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(1, 1), visualise=False)
  #feature = hog(resize, 2)
  print feature.shape
  return feature

fin = []
for number in range(1,11):
    output = chinese_filter(number)
    fin.append(output)
np.save('pei.npy', fin)
'''
print feature.shape
plt.figure()
plt.imshow(img)
plt.show()
for i in range(31):
 plt.imshow(feature[:,:,i])
 plt.show()

'''
