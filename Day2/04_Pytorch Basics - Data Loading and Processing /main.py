# Ref : http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Author : Sasank Chilamkurthy
# Editor : Junsik Choi

import os
import torch
import pandas as pd
import elice_utils
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from functions import show_landmarks, show_landmarks_batch
from dataset import FaceLandmarksDataset
from transforms import Rescale, RandomCrop, ToTensor
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


''' 01. Inspect Dataset '''
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.ix[n, 0]
landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
landmarks = landmarks.reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

show_landmarks(io.imread(os.path.join('./data/faces/', img_name)),landmarks)


''' 02. Instantiate dataset class and iterate through the data samples '''

face_dataset = FaceLandmarksDataset(csv_file='./data/faces/face_landmarks.csv',
                                    root_dir='./data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.savefig('figure.png')
        elice_utils.send_image('figure.png')
        plt.clf()
        break
        
''' 03. Transforms '''

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.savefig('figure.png')
elice_utils.send_image('figure.png')
plt.clf()

''' 04. Iterating through the dataset'''

transformed_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                           root_dir='faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

''' 05. Using Dataloader '''

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)

        break


