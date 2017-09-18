# Implementing Feed Forward Neural Network from scratch

NeuralNetwork.py에 구현된 NeuralNetwork class를 살펴봅시다.

Class NeuralNetwork의 내부 구조는 다음과 같습니다.

### class NeuralNetwork

#### Class Variables

- num_units : 모델의 전체 구조와 뉴런 수를 결정 [a,b,c]
- eta : 모델의 learning rate
- epochs : 모델의 학습 epoch 수를 결정
- weight_initialized : 모델 weight가 초기화 되었는가의 여부를 저장
- bShuffle : 데이터를 shuffling 할 것인가에 대한 여부를 결정
- minibatch_size : 모델의 학습에 사용될 minibatch 크기를 결정
- activation_function : 모델이 사용할 activation 함수를 결정
- verbose
- weights : 모델 각 층의 weight를 저장하는 list
- gradient : 모델의 gradient를 저장하는 list
- delta : 모델 backpropagation을 위한 delta 값을 저장하는 list
- net_value : 
- activation_value : 모델의 각 상태에서의 activation 값을 저장
- training_cost : 학습 과정에서 minibatch 별 모델의 평균 loss를 저장하는 list
- training_error : minibatch 별 Training error를 저장하는 list
- validation_error : minibatch 별 Valiation error를 저장하는 list

#### Methods
- buffer_clear : 현재 모델에 저장된 값들을 모두 비웁니다.
- initialize_weights : buffer_clear 함수를 이용, 모델을 비운 후, 모델 weight와 보조 변수들을 초기 값으로 설정합니다. (weight의 경우 정규분포 (평균:0, 표준편차:1)) 
- compute_loss : 모델의 마지막 output layer의 activation 값과 loss를 계산합니다. 
- activation : 모델 각 층의 activation 값을 계산합니다. 마지막 output 층의 경우 softmax 함수가 activation 함수로 작용합니다.
- forward : input - hidden ... hidden - output 층을 따라서 모델의 activation 값을 전달합니다.

##### Activation 함수와 Backpropagation을 위한 해당 함수들의 미분 함수의 정의
- sigmoid : Neuron의 sigmoid activation function
- tanh : Neuron의 hyperbolic tangent activation function
- softmax : 각 class label에 대한 확률을 계산
- derivative_activation : activation function의 미분 값을 반환
- derivative_sigmoid : sigmoid 함수의 미분 값을 반환
- derivative_tanh : tanh 함수의 미분 값을 반환

- predict : 각 입력에 대한 모델의 예측 값을 저장
- get_predicted_label : 모델이 predict 한 값을 반환

- train : 학습 데이터에 대해 epoch 만큼 학습을 수행. 한 epoch 안에서 
- compute_gradient : 각 Forward propagation에 대한 Back Propagation을 수행하여, gradient를 계산합니다.
- update_weights : 모델 weight를 learning rate에 따른 비율로 업데이트 시키는 함수. 
- vector_augmentation : activation 함수와 compute_gradient 함수를 수행하기 위한 helper function.

- shuffle : Data 순서를 섞어주는 helper function
- one_hot_coding : Data label을 one_hot coding 방식으로 변환시켜주는 함수
- plot_decision_regions : 모델의 Decision region을 plotting 하는 함수 (2D만 가능)