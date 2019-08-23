from __future__ import print_function
import sys
from flask import Flask,request,jsonify
from keras.models import load_model
import pandas as pd
import numpy as np
from flask_cors import CORS
import json
import base64
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib2
from skimage.transform import resize
from skimage.util import invert
app = Flask(__name__)
print('This is error output', file=sys.stderr)
print('This is standard output', file=sys.stdout)

data=pd.read_csv("k49_classmap.csv")
data.head()
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
def rgb2gray(rgb):

    r, g, b = rgb[:,:,3], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
@app.route('/img',methods= ['POST'])
def img():
	#content = request.get_json()
	#print(content,file=sys.stdout)
	#plt.waitforbuttonpress()

	url=str(json.loads(request.data))
	#print(url,file=sys.stdout)
	
	content = url.split(';')[1]
	image_encoded = content.split(',')[1]
	fh = open("download.png", "wb")
	fh.write(image_encoded.decode('base64'))
	fh.close()
	img=plt.imread("download.png")
	print(img.shape)
	img=resize(img, (28, 28))
	img=rgb2gray(img)
	##img=invert(img)
	img = (img * 500).astype(np.uint8)
	#print(img)
	img=np.expand_dims(img,axis=2)
	img=np.expand_dims(img,axis=0)
	#print(img.shape)
	arr=model.predict(img)

	ind=0
	#print(arr)
	for i in range(arr[0].shape[0]):
		if arr[0][i]>arr[0][ind]:
			ind=i
	predchar=data['char'][ind]
	print(predchar)
	
	#print(img)
	#img=m.imread(img,mode='I')
	#img=m.imresize(img, (28, 28))
	#img=np.invert(img)
	#arr=model.predict(img)

	#ind=0
	#print('Hello world!'+img.shape[0], file=sys.stderr)

	#for i in range(arr.shape[0]):
	#	if arr[i]>arr[ind]:
	#		ind=i
	#predchar=data['char'][ind]
	#return predchar
	return jsonify(predchar)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)