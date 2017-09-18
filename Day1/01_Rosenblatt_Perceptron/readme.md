## Rosenblatt's Perceptron

![Rosenblatt Perceptron Diagram](https://github.com/ku-milab/DeepLearningTutorial/blob/master/Data/figures/Rosenblatt_Perceptron_diagram.png?raw=true)

> Perceptron은 인공신경망의 한 종류로서, 1957년에 코넬 항공 연구소의 프랑크 로젠블라트 (Frank Rosenblatt)에 의해 고안되었습니다. 이것은 가장 간단한 형태의 피드포워드(Feedforward) 네트워크 - 선형분류기- 으로도 볼 수 있습니다. 

> Perceptron은 각 노드의 가중치(weight)와 입력치를 곱한 것을 모두 합한 값이 활성함수에 입력되어 동작됩니다. 보통 그 출력 값이 어떤 임계치 (보통 0)보다 크면 뉴런이 활성화되고, 결과값으로 1을 출력하며, 뉴런이 활성화되지 않으면 -1을 출력합니다.

![Iris dataset](http://blogfiles1.naver.net/20140605_53/kby990602l2_14019666846358H8hz_JPEG/%BA%D7%B2%C92.jpg)

> 실습에서 사용 할 데이터는 iris dataset 입니다. Iris dataset은 4가지 feature(꽃밪침 길이,너비/ 꽃잎 길이,너비)를 가지고, 50개씩 3가지 iris 종류로 총 150개의 데이터로 형성되어 있습니다. 이번 실습에선 첫 100개, 즉 2가지 iris만 사용 할 것입니다.