import enum
import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional
from torch.utils.data import DataLoader
import numpy as np
# from models.processing import *
# import segmentation_models_pytorch as smp
import torch.nn as nn
import matplotlib.image as mpimg

road = torch.tensor([64,32,32]).unsqueeze(0)
lane = torch.tensor([255,0,0]).unsqueeze(0)
undrive = torch.tensor([128,128,96]).unsqueeze(0)
moveable = torch.tensor([0,255,102]).unsqueeze(0)
mycar = torch.tensor([204,0,255]).unsqueeze(0)

colourmap = torch.cat((road,lane,undrive,moveable,mycar), dim =0)
colourmap = colourmap / 255

def mask_encode(img, colourmap, *args): #currently the dataloader is doing this for each image - should change this to do in batches
    #update later to pull the unique colors automatially
    
    pth = [arg for arg in args]
        # print(arg)
    assert(len(img.shape) == 3)
    _, x, y = img.shape
    out = torch.zeros(colourmap.shape[0], x, y)
    
    for i, rgb in enumerate(colourmap):
        # mask = (img[:,0] == rgb[0]) & (img[:,1] == rgb[1]) & (img[:,2] == rgb[2])
        # out[:,i][mask] = 1
        mask = (img[0] == rgb[0]) & (img[1] == rgb[1]) & (img[2] == rgb[2])
        out[i][mask] = 1
    return out

class ImageDataset(Dataset):

    CLASSES = ['road',
            'lane_markings',
            'undriveable',
            'moveable_obj',
            'MYcar']

    def __init__(self, images_dir, masks_dir, colourmap = colourmap, val = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.train_images = []#os.listdir(images_dir)
        self.val_images = []
        for name in os.listdir(imgdir):
            if name.endswith('9.png'):
                self.val_images.append(name)
            else:
                self.train_images.append(name)
        self.colourmap = colourmap
        self.padder = nn.ZeroPad2d((10,10,11,11))
        self.val = val


    def __getitem__(self, i):
        #read in images and masks
        if self.val:

            img_path = os.path.join(self.images_dir, self.val_images[i])
            mask_path = os.path.join(self.masks_dir, self.val_images[i])
        else:

            img_path = os.path.join(self.images_dir, self.train_images[i])
            mask_path = os.path.join(self.masks_dir, self.train_images[i])           
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = TF.to_tensor(image)
        mask = Image.open(mask_path)
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')
        mask = TF.to_tensor(mask)

        if mask.shape[0] == 4:
            mask = mask[:3,:,:,]   
             
        mask = mask_encode(mask, self.colourmap, mask_path)

        if image.shape != [1,3,896, 1184]:
            image = self.padder(image)
        if mask.shape != [1,5,896, 1184]:
            mask = self.padder(mask)
        assert(image.shape[-2:] == mask.shape[-2:] == (896,1184))
        
        return image, mask

    # def getValSet(self):
    #     '''
    #     valdir should just be a list containing file names 
    #     '''
    #     out_imgs = None
    #     out_masks = None
    #     for image in self.val_images:
    #         img_path = os.path.join(self.images_dir, image)
    #         mask_path = os.path.join(self.masks_dir, image)
    #         image = Image.open(img_path)
    #         if image.mode != 'RGB':
    #             image = image.convert('RGB')
    #         image = TF.to_tensor(image).unsqueeze(0)
    #         mask = Image.open(mask_path)
    #         if mask.mode != 'RGB':
    #             mask = mask.convert('RGB')
    #         mask = TF.to_tensor(mask).unsqueeze(0) 

    #         if out_imgs is None:
    #             out_imgs = image
    #         else:
    #             out_imgs = torch.cat((out_imgs, image)) 
    #         if out_masks is None:
    #             out_masks = mask
    #         else:
    #             out_masks = torch.cat((out_masks, mask))
    #     return out_imgs, out_masks

    def __len__(self):
        if self.val:
            return len(self.val_images)
        return len(self.train_images)



print(os.getcwd())

imgdir = 'comma10k/imgs'
maskdir = 'comma10k/masks'

train_filenames = []
val_filenames = []

# for name in os.listdir(imgdir):
    # if name.endswith('9.png'):
    #     val_filenames.append(name)
    # else:
    #     train_filenames.append(name)

train_dset = ImageDataset(imgdir, maskdir)#, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
