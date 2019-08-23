from __future__ import print_function
import sys
from flask import Flask,request
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from flask_cors import CORS
import json
from skimage.transform import resize
from skimage.util import invert
data=pd.read_csv("k49_classmap.csv")
data.head()
def rgb2gray(rgb):

    r, g, b = rgb[:,:,3], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
img=plt.imread("download.png")
print(img.shape)
img=resize(img, (28, 28))
img=rgb2gray(img)
#img=invert(img)
img = (img * 500).astype(np.uint8)
plt.imshow(img)
plt.waitforbuttonpress()
print(img)
img=np.expand_dims(img,axis=2)
img=np.expand_dims(img,axis=0)
print(img.shape)
arr=model.predict(img)

ind=0
print(arr)
for i in range(arr[0].shape[0]):
	if arr[0][i]>arr[0][ind]:
		ind=i
predchar=data['char'][ind]
print(predchar)