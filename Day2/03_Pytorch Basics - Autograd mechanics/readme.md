# Autograd mechanics

이번 예제에서는 Pytorch의 핵심적인 기능을 하는 `Autograd` 패키지에 대해 알아보겠습니다.

- `Autograd` 는 Pytorch를 이용하여 신경망을 학습 시키는데 가장 핵심적인 역할을 하는 Pytorch 패키지 입니다.
- `Autograd`는 텐서에 정의된 연산에 대한 미분(differentiation)을 제공합니다.
- 실행 후 정의(Define-by-run) 형식을 가지기 때문에, 동적인 그래프를 만들어, 네트워크의 그래프, Backpropagation의 경로가 각 iteration마다 유연하게 바뀔 수 있습니다.


## Variable
![Autograd_Variable](http://pytorch.org/tutorials/_images/Variable.png)

`autograd.Variable`은 `autograd` 패키지의 핵심 클래스입니다. `Variable` 클래스는 텐서를 감싸서, 우리가 Backpropagation을 위해  모델의 `backward()` 메소드를 호출했을 때,그 텐서에 대해 정의된 연산에 대한 Gradient 계산을 수행합니다.

- 텐서에 대한 데이터는 텐서의 `.data` 속성으로 접근 가능하고, `Variable`의 Gradient는 `.grad` 속성에 축적됩니다.


## Function

`autograd.Function`은 `Variable`과 함께 비순환 그래프(Acyclic graph)를 구성합니다. `Function` 클래스가 하는 일은 일어나는 계산의 순서를 인코딩 하는 것입니다. 각각의 `Variable`은 그 Variable을 만들어낸 Function을 가지고 있습니다. 이는 Variable의 `.grad_fn` 속성으로 접근할 수 있습니다.

## 미분값의 계산

`Variable`에 `.backward()`를 적용하여 해당 `Variable`에 대한 미분값을 계산할 수 있습니다. 