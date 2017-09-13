# License: BSD
# Author: Sasank Chilamkurthy
# Edit : Junsik choi

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
from functions import imshow, train_model, exp_lr_scheduler, visualize_model
'''
<Finetuning.py>

본 예제에서는 18 layer resnet을 로드하여 수행하고자 하는 Task(bee/ant classification)에 맞게 finetuning
하는 과정을 실습해 보겠습니다.

실습 코드는 크게 다음과 같이 이루어져있습니다

    - 데이터셋 로드 및 transform
    - 모델 로드 및 타겟 task에 맞게 변경
    - 모델 training 및 validation
    - 모델 예측 결과 visualize

'''

'''
1. Load data

torchvision 라이브러리의 transform을 통해서 훈련 데이터에 대해서 Crop, augmentation, Normalize를,
Validation 데이터에 대해서 Crop, Normalize를 편리하게 수행할 수 있습니다.
'''

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터 경로 및 data loader 설정
data_dir = 'data'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}

# 데이터 사이즈와 클래스
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

dset_classes = dsets['train'].classes

# GPU 사용 가능 여부
use_gpu = torch.cuda.is_available()

# Data Loader에서 batch size 만큼의 훈련 데이터 및 클래스 로드
inputs, classes = next(iter(dset_loaders['train']))

# 한 batch 안의 이미지들을 grid로 만들어서 확인해 보기
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[dset_classes[x] for x in classes])

'''
2. Model Load and Modify
'''

# Pretrain된 18 layer residual network를 로드
model_ft = models.resnet18(pretrained=True)
# 마지막 레이어의 입력 feature 수
print("Original Fully connected layer of resnet18 (Last layer):", model_ft.fc)
num_ftrs = model_ft.fc.in_features
# 마지막 레이어를 타겟 task에 맞게 수정
model_ft.fc = nn.Linear(num_ftrs, 2)
print("Modified Fully connected layer of resnet18 (Last layer):", model_ft.fc)


if use_gpu:
    model_ft = model_ft.cuda()
# Loss function 설정
criterion = nn.CrossEntropyLoss()

# 모델의 Optimizer 설정
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

'''
3. Train & Evaluate
'''
# model training
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       use_gpu, dset_loaders,dset_sizes,num_epochs=25)

'''
4. model visualize
'''
visualize_model(model_ft, dset_loaders, dset_classes, use_gpu)

plt.ioff()
plt.show()
