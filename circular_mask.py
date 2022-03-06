import cv2
import numpy as np
import os
from os.path import isfile, join
from os import listdir

directory = r'C:/Users/acer/Desktop/new_images_DeepSky/train'
os.chdir(directory)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


mypath='C:/Users/acer/Desktop/DeepSky/train'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  img = cv2.imread( join(mypath,onlyfiles[n]) )
  h, w = img.shape[:2]
  mask = create_circular_mask(h, w)
  masked_img = img.copy()
  masked_img[~mask] = 0
    
  filename = onlyfiles[n]
  
  cv2.imwrite(filename, masked_img)








