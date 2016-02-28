# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skimage.io as io
from skimage.color import rgb2gray
from skimage import img_as_float

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    print(codebook)
    print(labels)
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def MSE(orig_image, ch_image):
    w, h = orig_image.shape
    M = orig_image - ch_image
    mse = np.sum(M*M)/(w*h)
#    print('mse = %f') % mse
    return mse 

n_colors = 2

image = io.imread('parrots.jpg')
w, h, d = original_shape = tuple(image.shape)
print('w = %d h = %d d = %d') %(w,h,d)

gray_im = rgb2gray(image)
fl_im = img_as_float(image)
image_array = np.reshape(fl_im, (w * h, d))

kmeans = KMeans(n_clusters=n_colors, init='k-means++', random_state=241).fit(image_array)
labels = kmeans.predict(image_array)

new_im = recreate_image(kmeans.cluster_centers_, labels, w, h)
PSNR = 10*np.log10(1./MSE(rgb2gray(fl_im),rgb2gray(new_im)))

# Display all results, alongside original image
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image')
plt.imshow(fl_im)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (Clusters = %d, PSNR = %f)'%(n_colors,PSNR)) 
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.show()