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

print('here')
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
trainloader = DataLoader(train_dset, batch_size=8, shuffle=True, num_workers=12)

val_dset = ImageDataset(imgdir, maskdir, val=True)
valLoader = DataLoader(val_dset, batch_size=4, shuffle=True, num_workers=12)

num_epochs = 10
def train():
    print('starting the training loop . ... ... . .')
    for epoch in range(num_epochs):
        print(f'\nbegin epoch number {epoch + 1}')
        for index, (x, label) in enumerate(trainloader, 0):
            
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
                print('\nyou suck at machine learning')
        
        #validate at end of epoch
        print(f'\nvalidating at end of epoch {epoch +1}')
        model.eval()
        with torch.no_grad():
            torch.save(model.state_dict(), f'models/state_dicts/model_epoch{epoch +1}.pth')
            losses = []
            for index, (x, label) in enumerate(valLoader):
                x.to(DEVICE)
                label = label.to(DEVICE)

                y = model(x)

                loss = criterion(y, label)
                losses.append(loss.cpu().numpy())
            
            print(f'\naverage loss on epoch {epoch +1} was {np.mean(losses)}')
            print('continue training')
        model.train()


if __name__ == '__main__':
    train()
