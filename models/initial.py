import os
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.io import image as img
from torchvision.io import read_image
#from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import torchvision.transforms.functional as TF
import os
import torch.nn as nn
from dataloader import ImageDataset


ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 5
ACTIVATION = 'softmax'
DEVICE = torch.device('cuda')

model = smp.Unet(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    in_channels = 3,
    classes = CLASSES,
    activation=ACTIVATION).to(DEVICE)

model = torch.nn.DataParallel(model)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam([ 
    dict(params=model.module.parameters(), lr=0.0001),
])

# impath = 'models/data/testmask.png'
# img = Image.open(impath)
# x = TF.to_tensor(img).to(DEVICE)
# x = x.unsqueeze(0)
# print(x.shape)


assert(True)
imgdir = 'comma10k/imgs'
maskdir = 'comma10k/masks'

train_dset = ImageDataset(imgdir, maskdir)#, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
trainloader = DataLoader(train_dset, batch_size=6, shuffle=False, num_workers=12)

num_epochs = 2
def train():
    print('starting the training loop . ... ... . .')
    for epoch in range(num_epochs):
        print(f'\nbegin epoch number {epoch + 1}')
        for index, (x, label) in enumerate(trainloader, 0):
            print(index)
            #right now we have x.shape == (4,1,3,896)
            #please fix this later
            # assert(x.shape == (4,1,3,896, 1184))
            #has been fixed
            # if x.shape != (4,3,896,1184):
            #     print(f'shape of x is {x.shape} please do something about this')
            # # x, label = x.squeeze(1), label.squeeze(1)
            
            model.zero_grad()

            x = x.to(DEVICE)
            label = label.to(DEVICE)

            #fwd pass
            y = model(x)

            #calc loss
            realLoss = criterion(y, label)
            realLoss.backward()
            optimizer.step()

            if (index + 1)% 20 ==0:
                print(f'loss is currently {realLoss}')
                print('\n you suck at machine learning')

if __name__ == '__main__':
    train()
