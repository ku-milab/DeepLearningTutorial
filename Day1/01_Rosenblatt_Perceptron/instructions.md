## Instructions

- 본 실습에서는 가장 기본적인 형태의 퍼셉트론인 Rosenblatt Perceptron의 핵심 기능을 직접 구현해 보고 학습시키며, 그 학습 결과와 Decision Region을 시각화해 보겠습니다.

##### Task 1
- `Perceptron_Rosenblatt.py` 의 `Class Perceptron`의 method중 하나인 `net_input`을 구현하세요.

##### Task 2
- 동일한 파일에서 `Class Perceptron`의 method `predict`을 구현하세요.

##### Task 3
- 동일한 파일에서 `Class Perceptron`의 method `fit`을 구현하세요.

##### Task 4
- `main.py`에서 첫번째 함수 `check_and_load_dataset()` 만을 실행시킨 후, 학습 데이터를 확인하세요.

##### Task 5
- `main.py`에서 두번째 함수 `training_Perceptron(X,y,learning_rate,num_epochs)`를 확인하고, `Perceptron_Rosenblatt.py` 에서 정의한 `Class Perceptron`을 이용하여 classifier를 생성한 후 학습을 실행하세요. learning rate와 epoch를 바꿔가며 학습을 진행시켜 보세요.

##### Task 5
- `main.py`에서 세번째 함수 `Plot_decision_region(X,y,classifier)`를 실행시켜 학습한 Perceptron의 Decision Region을 확인해보세요.