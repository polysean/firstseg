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
    for arg in args:
        pass
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

    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None, colourmap = colourmap):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(images_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.colourmap = colourmap
        self.padder = nn.ZeroPad2d((10,10,11,11))


    def __getitem__(self, i):
        #read in images and masks
        # print('get')
        img_path = os.path.join(self.images_dir, self.images[i])
        mask_path = os.path.join(self.masks_dir, self.images[i])

        print(img_path[13:] == mask_path[14:])
        image = Image.open(img_path)
        image = TF.to_tensor(image)
        mask = Image.open(mask_path)
        mask = TF.to_tensor(mask)

        # if len(image.shape) == 3:
        #     image =image.unsqueeze(0)
        if mask.shape[0] == 4:
            mask = mask[:3,:,:,]   
              
        mask = mask_encode(mask, self.colourmap, mask_path)

        if image.shape != [1,3,896, 1184]:
            image = self.padder(image)
        if mask.shape != [1,5,896, 1184]:
            mask = self.padder(mask)
        assert(image.shape[-2:] == mask.shape[-2:] == (896,1184))

        return image, mask


    def __len__(self):
        return len(self.images)


print(os.getcwd())

imgdir = 'comma10k/imgs'
maskdir = 'comma10k/masks'

# ENCODER = 'efficientnet-b3'
# ENCODER_WEIGHTS = 'imagenet'
# CLASSES = 16
# ACTIVATION = 'softmax'
# DEVICE = torch.device('cuda')

# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dset = ImageDataset(imgdir, maskdir)#, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
trainloader = DataLoader(train_dset, batch_size=4, shuffle=True, num_workers=12)

x = train_dset.__getitem__(0)
assert(True)