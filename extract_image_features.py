import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
import time
import argparse

train = pd.read_csv('features-validation.csv')
img_size = 256
batch_size = 256

ap = argparse.ArgumentParser()
ap.add_argument('-worker', type=int, required=True)
args = ap.parse_args()
worker = args.worker

ids = train['id'].values
del train
n_batches = len(ids) // batch_size + 1
total_workers = 20

print ("Total Batches: " + str(n_batches))
batches_per_worker = n_batches // 20
start_batch = worker * batches_per_worker
final_batch = (worker + 1) * batches_per_worker
if worker + 1 == total_workers:
	final_batch = n_batches + 1
print(range(start_batch, final_batch))

from keras.applications.nasnet import preprocess_input, NASNetMobile


def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path, pet_id):
    image = cv2.imread(path + str(pet_id) + '.jpg')
    new_image = resize_to_square(image).astype(np.float32)
    new_image = preprocess_input(new_image)
    return new_image


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = NASNetMobile(input_tensor = inp, include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)
f = open("validation-images-features_" + str(worker) + ".csv", "a+")

for b in tqdm(range(start_batch, final_batch)):
    a = time.time()
    start = b*batch_size
    end = (b+1)*batch_size
    batch_ids = ids[start:end]
    batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
    b = time.time()
    print ("Configuring Batches: " + str(b-a))
    for i, house_id in enumerate(batch_ids):
        try:
            batch_images[i] = load_image("images-validation/", house_id)
        except: 
            pass
    c = time.time()
    print ("Processing Images: " + str(c - b))
    batch_preds = m.predict(batch_images, batch_size = batch_size)
    d = time.time()
    print ("Running Network: " + str(d - c))
    for i,house_id in enumerate(batch_ids):
    	feature = ", ".join(map(str, batch_preds[i]))
    	row = str(house_id) + ", " + feature + "\n"
    	f.write(row)
    e = time.time()
    print ("Writing to file " + str(e - d))
