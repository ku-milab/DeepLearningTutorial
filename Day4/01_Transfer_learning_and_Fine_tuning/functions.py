# License: BSD
# Author: Sasank Chilamkurthy
# Edit : Junsik choi
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

'''
Visualize Images
'''

def imshow(inp, title=None):
    """Imshow for Tensor."""
    # Channel x Height x Width -> Height x Width x Channel
    inp = inp.numpy().transpose((1, 2, 0))

    # Recover normalized image to original mean and standard variation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean

    # Show
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # pause a bit so that plots are updated
    plt.pause(0.001)

# Add elice_utile func


'''
Training
'''
def train_model(model, criterion, optimizer, lr_scheduler, use_gpu, dset_loaders, dset_sizes, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 epoch 별로 train phase와 validation phase를 진행
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # 모델을 train 모드로 설정
            else:
                model.train(False)  # 모델을 validation 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # dset_loader의 데이터를 순차적으로 로드
            for data in dset_loaders[phase]:
                # input, label 로드
                inputs, labels = data

                # inputm label을 Variable로 wrapping
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # optimizer의 gradient를 0으로 초기화
                optimizer.zero_grad()

                # input을 모델에 forward / 모델 prediction 도출 / loss 계산
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Training phase에서 backpropagation 진행
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 모델 loss와 성공적으로 Prediction 수를 저장
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            # epoch 별 평균 loss와 accuracy를 계산
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # validation phase에서 epoch 별 가장 좋은 성능을 내고 있는 모델을 저장
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_model

'''
Learning rate scheduler
매 epoch 마다 learning rate를 1/10 으로 감소
'''
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    # optimize 시키고 있는 parameter의 learning rate를 변경
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

''' Visualizing the model predictions '''

def visualize_model(model, dset_loaders, dset_classes,use_gpu, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dset_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return
