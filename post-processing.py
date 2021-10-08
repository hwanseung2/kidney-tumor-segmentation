import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import logging
import os
import sys
import tempfile
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from pathlib import Path
import time
from PIL import Image
import pydicom
import random

import monai
from monai.optimizers import LearningRateFinder
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
import monai.transforms as mtf
from monai.networks import one_hot

arch = "resnet50" # arch : "resnet34", "eff-b0", "eff-b5"
wl, ww = 112, 384

submission_name = "submission30.csv"

root_dir = "./body-morphometry-kidney-and-tumor"
pth_dir = "./body-morphometry-kidney-and-tumor/pth/"

# ## Pre-train Positive_only

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars\n
        torch.backends.cudnn.deterministic = True  #needed\n
        torch.backends.cudnn.benchmark = False
        
seed = 42
random_seed(seed,True)

def window_image(image, window_center, window_width): #윈도윙 해주기
  img_min = window_center - window_width // 2
  img_max = window_center + window_width // 2
  window_image = image.copy()
  window_image[window_image < img_min] = img_min
  window_image[window_image > img_max] = img_max
  return window_image

def bwperim(bw, n=4):

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

def signed_bwdist(im):
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    #im = im - scipy.ndimage.morphology.binary_dilation(im)
    return im

def bwdist(im):
    dist_im = scipy.ndimage.distance_transform_edt(1-im)
    return dist_im

def interp_shape(top, bottom, precision):
    if precision>1:
        print("Error: Precision must be between 0 and 1 (float)")

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids 
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r**2, 2))
    xi = np.c_[np.full((r**2),precision), xi]

    # Interpolate for new plane
    out = scipy.interpolate.interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out
def OneHotEncode(image):
    lbl1 = (image==1)
    lbl2 = (image==2)
    return lbl1, lbl2

def RemoveSmallObjects(image):
    kidney,tumor = OneHotEncode(image)
    removed_kidney = morphology.remove_small_objects(kidney,100)
    removed_tumor = morphology.remove_small_objects(tumor,100)
    removed_kidney = removed_kidney.astype(np.uint8)
    removed_tumor = 2*removed_tumor.astype(np.uint8)
    removed_img = np.maximum(removed_kidney,removed_tumor)
    return removed_img
    
def RemoveFP(whole):
    kidney,tumor = OneHotEncode(whole)
    for z in range(1,63):
        k = np.count_nonzero(kidney[z,:,:])
        t = np.count_nonzero(tumor[z,:,:])
        if z!=0 and z!=63:
            k2 = np.count_nonzero(kidney[z-1,:,:])
            t2 = np.count_nonzero(tumor[z-1,:,:])
            k3 = np.count_nonzero(kidney[z+1,:,:])
            t3 = np.count_nonzero(tumor[z+1,:,:])

        if z==1:
            k4 = np.count_nonzero(kidney[z+2,:,:])
            t4 = np.count_nonzero(tumor[z+2,:,:])
            if k!=0 and (k2+k3+k4)==0:
                kidney[z,:,:] = 0
            if t!=0 and (t2+t3+t4)==0:
                tumor[z,:,:] = 0
        elif z==62:
            k1 = np.count_nonzero(kidney[z-2,:,:])
            t1 = np.count_nonzero(tumor[z-2,:,:])
            if k!=0 and (k1+k2+k3)==0:
                kidney[z,:,:] = 0
            if t!=0 and (t1+t2+t3)==0:
                tumor[z,:,:] = 0
        else:
            k1 = np.count_nonzero(kidney[z-2,:,:])
            t1 = np.count_nonzero(tumor[z-2,:,:])
            k4 = np.count_nonzero(kidney[z+2,:,:])
            t4 = np.count_nonzero(tumor[z+2,:,:])
            if k!=0 and (k1+k2+k3+k4)==0:
                kidney[z,:,:] = 0
            if t!=0 and (t1+t2+t3+t4)==0:
                kidney = kidney.astype(np.uint8)
                tumor = tumor.astype(np.uint8)
                a = (tumor[z,:,:] == 1)
                kidney[z,:,:] = kidney[z,:,:] + a*1
                tumor[z,:,:] = 0
    kidney = kidney.astype(np.uint8)
    tumor = 2*tumor.astype(np.uint8)
    removed_img = np.maximum(kidney,tumor)
    return removed_img

def InterpolateFN(whole):
    kidney,tumor = OneHotEncode(whole)
    for z in range(2,62):
        k = np.count_nonzero(kidney[z,:,:])
        k1 = np.count_nonzero(kidney[z-2,:,:])
        k2 = np.count_nonzero(kidney[z-1,:,:])
        k3 = np.count_nonzero(kidney[z+1,:,:])
        k4 = np.count_nonzero(kidney[z+2,:,:])
        t = np.count_nonzero(tumor[z,:,:])
        t1 = np.count_nonzero(tumor[z-2,:,:])
        t2 = np.count_nonzero(tumor[z-1,:,:])
        t3 = np.count_nonzero(tumor[z+1,:,:])
        t4 = np.count_nonzero(tumor[z+2,:,:])
        if k==0 and k1!=0 and k2!=0 and k3!=0 and k4!=0:
            kidney[z,:,:] = 1*interp_shape(kidney[z-1,:,:],kidney[z+1,:,:],0.5)
        if t==0 and t1!=0 and t2!=0 and t3!=0 and t4!=0:
            tumor[z,:,:] = 1*interp_shape(tumor[z-1,:,:],tumor[z+1,:,:],0.5)
    kidney = kidney.astype(np.int8)
    tumor = 2*tumor.astype(np.int8)
    interpolated_img = np.maximum(kidney,tumor)
    return interpolated_img