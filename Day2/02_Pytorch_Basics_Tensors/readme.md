## 1. 텐서(Tensors)

![Pytorch Logo](http://andersonjo.github.io/assets/posts2/Pytorch/logo.jpg)

- Pytorch에서 텐서란 고차원 데이터의 연산을 위한 데이터타입으로, Numpy의 ndarray와 기본적으로 같은 개념입니다.

- Pytorch Tensor 데이터는 CPU와 GPU 연산에 모두 사용될 수 있습니다. cuda 데이터타입으로 변환시키면, GPU의 고속 연산에 쓰일 수 있습니다.

- Pytorch의 텐서를 사용하여 GPU로 병렬 연산을 수행하면,CPU 연산보다 약 50배 정도 빠른 속도로 계산을 적용할 수 있습니다.

- 매우 많은 연산량을 요구하는 딥러닝 모델 학습에서는 GPU 연산이 필수적이므로, Pytorch에서는 GPU 연산에 적용 가능한 Tensor 데이터 타입을 사용합니다. 

