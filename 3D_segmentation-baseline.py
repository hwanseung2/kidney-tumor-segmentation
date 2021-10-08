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
import torch
from torch.utils.data import Dataset#, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
import pydicom
import random

import monai
from monai.optimizers import LearningRateFinder
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
import monai.transforms as mtf
from monai.networks import one_hot
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, UNet
from monai.networks.layers import Norm
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

num_workers = 6
batch_size = 2
pre_valid = range(81,101)
roi_size = (224,224,32)
wl, ww = 112,384
norm_mean, norm_std = 101, 76.9
learning_rate = 0.02
submission_name = "submission34.csv"
root_dir = "./body-morphometry-kidney-and-tumor"
pth_dir = "./body-morphometry-kidney-and-tumor/pth/"

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


def path_list(root_dir, mode, valid_num):
    data_list = []
    if mode == "train":
        for patient in tqdm(sorted(os.listdir(os.path.join(root_dir, "train/DICOM")))):
            if 'training' in patient and patient.split('g')[-1] not in valid_num:
                for z_slice in sorted(os.listdir(os.path.join(root_dir, "train/DICOM", patient))):
                    if z_slice.split('.')[-1] == "dcm":
                        name = z_slice.split('.')[0]
                        path_image = os.path.join(root_dir, "train/DICOM", patient, f"{name}.dcm")
                        path_label = os.path.join(root_dir, "train/Label", patient, f"{name}.png")
                        image_tmp = pydicom.dcmread(path_image).pixel_array[:,:,np.newaxis]
                        label_tmp = cv2.imread(path_label)[:,:,0]
                        label_tmp = label_tmp[:,:,np.newaxis]
                        if int(name) == 0:
                            cat_image = image_tmp
                            cat_label = label_tmp
                        else:
                            cat_image = np.concatenate((cat_image, image_tmp), axis = 2)
                            cat_label = np.concatenate((cat_label, label_tmp), axis = 2)
                            
                cat_image = window_image(cat_image, wl, ww)
                cat_image = (cat_image - norm_mean) / norm_std
#                 print(f"image shape : {cat_image.shape}, label shape : {cat_label.shape}")
                case = {"image" : cat_image, "label" : cat_label}
                data_list.append(case)
                            
                
    else:
        for patient in tqdm(sorted(os.listdir(os.path.join(root_dir, "train/DICOM")))):
            if 'training' in patient and patient.split('g')[-1] in valid_num:
                for z_slice in sorted(os.listdir(os.path.join(root_dir, "train/DICOM", patient))):
                    if z_slice.split('.')[-1] == "dcm":
                        name = z_slice.split('.')[0]
                        path_image = os.path.join(root_dir, "train/DICOM", patient, f"{name}.dcm")
                        path_label = os.path.join(root_dir, "train/Label", patient, f"{name}.png")
                        image_tmp = pydicom.dcmread(path_image).pixel_array[:,:,np.newaxis]
                        label_tmp = cv2.imread(path_label)[:,:,0]
                        label_tmp = label_tmp[:,:,np.newaxis]
                        if int(name) == 0:
                            cat_image = image_tmp
                            cat_label = label_tmp
                        else:
                            cat_image = np.concatenate((cat_image, image_tmp), axis = 2)
                            cat_label = np.concatenate((cat_label, label_tmp), axis = 2)
                            
                cat_image = window_image(cat_image, wl, ww)
                cat_image = (cat_image - norm_mean) / norm_std
                case = {"image" : cat_image, "label" : cat_label}
                data_list.append(case)
    return data_list


valid_num = []
for i in pre_valid:
    num = format(i, '03')
    valid_num.append(num)
train_list = path_list(root_dir, "train", valid_num)
valid_list = path_list(root_dir, "valid", valid_num)
print(f"train list : {len(train_list)}, valid list : {len(valid_list)}")


train_aug = mtf.Compose([
    mtf.AddChanneld(keys=["image", "label"]),
    mtf.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=3,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
    mtf.RandGaussianNoised(
        keys = ["image"],
        mean = 0.0,
        std = 0.1,
        prob = 0.2
    ),
    mtf.RandBiasFieldd(
        keys = ["image"],
        prob = 0.2
        ),
    mtf.RandGaussianSharpend(
        keys = ["image"],
        prob = 0.1
    ),
    mtf.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
    mtf.RandFlipd(
        keys=["image", "label"],
        spatial_axis=[1],
        prob=0.10,
    ),
    mtf.RandFlipd(
        keys=["image", "label"],
        spatial_axis=[2],
        prob=0.10,
    ),
    mtf.RandRotate90d(
        keys=["image", "label"],
        prob=0.10,
        max_k=3,
    ),
    mtf.RandAffined(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=0.1,
    spatial_size=roi_size,
    shear_range = (0.2,0.2,0.2),
    translate_range=(15, 15, 2),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="zeros",
    ),
    
#     mtf.Rand3DElasticd(
#     keys=["image", "label"],
#     mode=("bilinear", "nearest"),
#     prob=1,
#     sigma_range=(9, 12),
#     magnitude_range=(50, 100),
#     spatial_size=(224,224,16),
#     translate_range=(30, 30, 2),
#     scale_range=(0.15, 0.15, 0.15),
#     padding_mode="border",
#     ),
    mtf.ToTensord(keys=["image", "label"]),
    mtf.EnsureType(),
])
valid_aug = mtf.Compose([
    mtf.AddChanneld(keys=["image", "label"]),
    mtf.ToTensord(keys=["image", "label"]),
    mtf.EnsureType(),
])


train_ds = monai.data.Dataset(train_list, transform = train_aug)
val_ds = monai.data.Dataset(valid_list, transform = valid_aug)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = UNETR(
#     in_channels=1,
#     out_channels=3,
#     img_size=roi_size,
#     feature_size=8,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# loss_function = monai.losses.DiceCELoss(include_background = False, to_onehot_y=True, softmax=True)
loss_function = monai.losses.DiceCELoss(include_background=False, softmax = True, to_onehot_y = True, ce_weight = torch.tensor([0.3, 0.3, 0.4]).cuda())#, ce_weight = torch.tensor([0.25, 0.3,0.45]).cuda()
# loss_function = monai.losses.DiceLoss(include_background=False, softmax = True, to_onehot_y = True)
post_label = mtf.AsDiscrete(to_onehot=True, n_classes=3)
post_pred = mtf.AsDiscrete(argmax=True, to_onehot=True, n_classes=3)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

torch.backends.cudnn.benchmark = True
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda().float(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, roi_size, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda().float(), batch["label"].cuda())
#         print(x.shape, y.shape, x.dtype, y.dtype)
#         print
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                ppath = Path(os.path.join(pth_dir, f"{submission_name.split('.')[0]}/pre-best_model.pth"))
                ppath.parent.mkdir(parents = True, exist_ok = True)
                torch.save(model.state_dict(), str(ppath))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 150000
eval_num = 300
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )



