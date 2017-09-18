## Instructions

- 본 실습에서는 Rosenblatt Perceptron의 error function을 변형하여 Widrow-Hoff Perceptron을 직접 구현해 보겠습니다.

##### Task 1
- "Perceptron_Widrow_Hoff.py" 파일에서 `Class AdalineGD`의 method `fit`을 구현하세요.

##### Task 2
- `main.py`에서 첫번째 함수 'train_adaline_diff_lr(X, y)'를 확인하고, learning rate 또는 epoch를 바꿔가며 학습을 진행시켜 보세요.

##### Task 3
- `main.py`에서 두번째 함수 `data_standarization(X,y)`를 확인하고, 정규화 된 데이터를 사용해 학습을 진행시켜 보세요.

##### Task 4
- `main.py`에서 세번째 함수 `Plot_decision_regions(X,y,classifier)`를 실행시켜 학습한 Perceptron의 Decision Region을 확인해보세요.