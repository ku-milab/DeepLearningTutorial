## Transfer Learning Tutorial

컴퓨터 비전 분야에서, 컨볼루션 신경망(CNN)은 Image Classification, Detection, Segmentation 등 여러가지 Task에 대하여 강력한 성능(인간 이상)을 발휘합니다.

인터넷에는 LeNet, ResNet, 등과 같이 ImageNet과 같은 대형 이미지 데이터셋을 이미 학습한 여러가지 CNN 모델이 존재하는데,

이런 기 학습된 모델을 사용하기 위해 매번 대형 데이터셋에 대해서 초기부터 학습을 진행시켜야 한다면 많은 시간과 비용이 들어갈 것입니다.

다행히, Transfer Learning을 잘 응용하면 자신이 원하는 어떤 Task에 기 학습된 모델을 이런 모델을 초기(Scratch)단계부터 다시 학습시켜야 할 필요는 없습니다.

이미 학습된 모델의 Feature나 Knowledge를 Target Domain과 Task에 이용 가능하게 하는 Transfer Learning의 방법에는 여러가지가 있습니다.

