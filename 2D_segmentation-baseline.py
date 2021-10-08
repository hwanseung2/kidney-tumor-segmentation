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
from torch.utils.data import Dataset, DataLoader
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
from monai.data import decollate_batch
import albumentations as A
from albumentations.pytorch import ToTensorV2



arch = "resnet50" # arch : "resnet34", "eff-b0", "eff-b5"
random_windowing = True # random windowing : True - random, False - simple windowing(wl : 40, ww : 400)
pre_valid, fine_valid = range(81,101), range(81,101)
wl, ww = 112,384
batch_size = 8
num_workers = 6
num_epochs = 100
# lr = 0.004

submission_name = "submission35.csv"

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


def path_list(root_dir, mode, valid_num, positive_only = False):
    path_images = []
    path_labels = []
    if mode == "train":
        for patient in tqdm(sorted(os.listdir(os.path.join(root_dir, "train/DICOM")))):
            if 'training' in patient and patient.split('g')[-1] not in valid_num:
                for z_slice in sorted(os.listdir(os.path.join(root_dir, "train/DICOM", patient))):
                    if z_slice.split('.')[-1] == "dcm":
                        name = z_slice.split('.')[0]
                        path_image = os.path.join(root_dir, "train/DICOM", patient, f"{name}.dcm")
                        path_label = os.path.join(root_dir, "train/Label", patient, f"{name}.png")
                        if positive_only:
                            temp_label = cv2.imread(path_label)
                            if np.all(temp_label == 0):
                                continue
                            else:
                                path_images.append(path_image)
                                path_labels.append(path_label)
                        else:
                            path_images.append(path_image)
                            path_labels.append(path_label)
                            
                
    else:
        for patient in tqdm(sorted(os.listdir(os.path.join(root_dir, "train/DICOM")))):
            if 'training' in patient and patient.split('g')[-1] in valid_num:
                for z_slice in sorted(os.listdir(os.path.join(root_dir, "train/DICOM", patient))):
                    if z_slice.split('.')[-1] == "dcm":
                        name = z_slice.split('.')[0]
                        path_image = os.path.join(root_dir, "train/DICOM", patient, f"{name}.dcm")
                        path_label = os.path.join(root_dir, "train/Label", patient, f"{name}.png")
                        if positive_only:
                            temp_label = cv2.imread(path_label)
                            if np.all(temp_label == 0):
                                continue
                            else:
                                path_images.append(path_image)
                                path_labels.append(path_label)
                        else:
                            path_images.append(path_image)
                            path_labels.append(path_label)
    return path_images, path_labels

valid_num = []
for i in pre_valid:
    num = format(i, '03')
    valid_num.append(num)
train_images, train_labels = path_list(root_dir, "train", valid_num, False)
valid_images, valid_labels = path_list(root_dir, "valid", valid_num, False)
print(f"train images : {len(train_images)}, train labels : {len(train_labels)}")
print(f"valid images : {len(valid_images)}, valid labels : {len(valid_labels)}")

class MyDataset_train(Dataset):
    def __init__(self, images_list, labels_list, transform = None):
        self.images_list = images_list
        self.labels_list = labels_list
        self.transform = transform
    def __getitem__(self, idx):
        image = pydicom.dcmread(self.images_list[idx]).pixel_array
        if random_windowing:
            image = window_image(image,random.randrange(20,170), random.randrange(350,800))
        else:
            image = window_image(image, wl, ww)
#         image += abs(image.min())
#         image = image.astype(np.float32)
#         image = image / (image.max())
        mask = cv2.imread(self.labels_list[idx])[:,:,0]
        

        if self.transform:
            augmented = self.transform(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask[np.newaxis,:,:]
        return image, mask
    def __len__(self):
        return len(self.images_list)

class MyDataset_valid(Dataset):
    def __init__(self, images_list, labels_list, transform = None):
        self.images_list = images_list
        self.labels_list = labels_list
        self.transform = transform
    def __getitem__(self, idx):
        image = pydicom.dcmread(self.images_list[idx]).pixel_array
        image = window_image(image, wl, ww)
#         image += abs(image.min())
#         image = image.astype(np.float32)
#         image = image / (image.max())
        mask = cv2.imread(self.labels_list[idx])[:,:,0]
        

        if self.transform:
            augmented = self.transform(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask[np.newaxis,:,:]
        return image, mask
    def __len__(self):
        return len(self.images_list)

def strong_aug(p=0.5):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
        ], p=0.2),
        A.OneOf([
            A.Sharpen(),
            A.RandomBrightnessContrast(),
            A.ElasticTransform()
        ], p=0.3),
        A.OneOf([
            A.Affine(scale = (1.2, 0.8)),
            A.Affine(translate_percent = (0.2, 0.2)),
            A.Affine(rotate = (-30, 30)),
            A.Affine(shear = (-20,20)),
            A.GridDistortion(),
            A.OpticalDistortion(),
        ], p = 0.8),
    ], p=p)
aug_func = strong_aug(p = 0.9)


train_aug = A.Compose([
    A.Normalize(mean = 101, std = 76.9, max_pixel_value = 1),
    aug_func,
    ToTensorV2(),
])

valid_aug = A.Compose([
    A.Normalize(mean = 101, std = 76.9, max_pixel_value = 1),
    ToTensorV2(),
])
trainset = MyDataset_train(train_images, train_labels, transform = train_aug)
validset = MyDataset_valid(valid_images, valid_labels, transform = valid_aug)
    
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers = num_workers)
valid_loader = DataLoader(validset, batch_size=batch_size, shuffle = False, num_workers = num_workers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Architecture
"""
if arch == "resnet34":
    model = smp.Unet(encoder_name = "resnet34", encoder_weights = "imagenet", in_channels = 1, classes = 3)
elif arch == "resnet50":
    model = smp.Unet(encoder_name = "resnet50", encoder_weights = "imagenet", in_channels = 1, classes = 3)
elif arch == "resnet101":
    model = smp.Unet(encoder_name = "resnet101", encoder_weights = "imagenet", in_channels = 1, classes = 3)
elif arch == "eff-b5":
    model = smp.Unet(encoder_name = "timm-efficientnet-b5", encoder_weights = "noisy-student",in_channels = 1, classes = 3)
elif arch == "eff-b0":
    model = smp.Unet(encoder_name = "timm-efficientnet-b0", encoder_weights = "noisy-student",in_channels = 1, classes = 3)


model = model.to(device)
# loss_function = monai.losses.DiceCELoss(include_background=False, softmax = True, to_onehot_y = True, ce_weight = torch.tensor([0.3, 0.3,0.4]).cuda())
loss_function = monai.losses.GeneralizedDiceLoss(include_background=False, softmax = True, to_onehot_y = True)
post_trans = mtf.Compose(
    [mtf.Activations(softmax=True)]
)
post_label = mtf.AsDiscrete(to_onehot=True, n_classes=3)
post_pred = mtf.AsDiscrete(argmax=True, to_onehot=True, n_classes=3)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
inferer = monai.inferers.SimpleInferer()


# # In[26]:

lower_lr, upper_lr = 1e-5, 1e-0
optimizer = torch.optim.Adam(model.parameters(), lower_lr)
lr_finder = LearningRateFinder(model, optimizer, loss_function, device=device)
lr_finder.range_test(train_loader, valid_loader, end_lr=upper_lr, num_iter=20)
steepest_lr, _ = lr_finder.get_steepest_gradient()



print(steepest_lr)

optimizer = torch.optim.Adam(model.parameters(), steepest_lr)


val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

for epoch in range(num_epochs):
        
    print("-" * 10)
    print(f"epoch {epoch}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in tqdm(train_loader):
        step += 1
        inputs, labels = batch_data[0].to(device).float(), batch_data[1].to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_len = len(trainset) // train_loader.batch_size
        if step % 100 == 0:
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}",flush = True)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            val_images = None
            val_labels = None
            val_outputs = None
            val_step = 0
            val_epoch_loss = 0
            dice_vals = list()
            
            for val_data in tqdm(valid_loader):
                val_step += 1
                val_images, val_labels = val_data[0].to(device).float(), val_data[1].to(device).long()
                val_outputs = model(val_images)
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
#                 val_loss = loss_function(val_outputs, val_labels)
#                 val_epoch_loss += loss.item()
#                 val_epoch_len = len(validset) // valid_loader.batch_size
            
            metric = np.mean(dice_vals)
#             val_epoch_loss /= val_step
#             metric = val_epoch_loss
#             metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                ppath = Path(os.path.join(pth_dir, f"{submission_name.split('.')[0]}/pre-best_model.pth"))
                ppath.parent.mkdir(parents = True, exist_ok = True)
                torch.save(model.state_dict(), str(ppath))
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
    if epoch > 75:
        ppath = Path(os.path.join(pth_dir, f"{submission_name.split('.')[0]}/best_model_e{epoch}.pth"))
        ppath.parent.mkdir(parents = True, exist_ok = True)
        torch.save(model.state_dict(), str(ppath))
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")




