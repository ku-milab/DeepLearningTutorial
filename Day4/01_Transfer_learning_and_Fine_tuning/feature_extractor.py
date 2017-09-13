# License: BSD
# Author: Sasank Chilamkurthy
# Edit : Junsik choi

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from functions import imshow, train_model, exp_lr_scheduler, visualize_model

'''
<feature_extractor.py>

본 실습에서는 모델 파라미터를 학습된 상태에서 변하지 않도록 고정시켜 기학습된 모델을 Feature extractor로 사용할 것입니다.
Pytorch 에서는 모델의 파라미터에 대해서 requires_grad를 False로 설정하면 backward()과정에서 gradient가 계산되지 않습니다
finetuning과는 다르게 기학습된 모델을 feature extractor로 사용할 것이기 때문에, 마지막 레이어를 제외한 모델의 파라미터는 변하지 않게 고정합니다

실습 코드는 크게 다음과 같이 이루어져있습니다

    - 데이터셋 로드 및 transform
    - 모델 로드 및 타겟 task에 맞게 변경
    - 모델 training 및 validation
    - 모델 visualize

'''

def feature_extract():
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
    model_conv = torchvision.models.resnet18()
    model_conv.load_state_dict(torch.load('./data/models/resnet18-5c106cde.pth'))
    # Load한 모델의 파라미터를 train 과정에서 변경하지 않도록 설정 (requires_grad = False)
    for param in model_conv.parameters():
        param.requires_grad = False

    # 마지막 레이어를 타겟 task에 맞게 수정
    print("Original Fully connected layer of resnet18 (Last layer):", model_conv.fc)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    print("Modified Fully connected layer of resnet18 (Last layer):", model_conv.fc)

    if use_gpu:
        model_conv = model_conv.cuda()

    # Loss function 설정
    criterion = nn.CrossEntropyLoss()

    # 모델의 Optimizer 설정

    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    '''
    3. Train & Evaluate
    '''

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                           use_gpu, dset_loaders,dset_sizes,num_epochs=25)

    '''
    4. model visualize
    '''
    visualize_model(model_conv, dset_loaders, dset_classes, use_gpu)

    plt.ioff()
    plt.show()

    return
