import elice_utils
import torch
import numpy as np

def main():

    examples_create_tensor()
    examples_manipulate_tensor()
    examples_operate_tensor()
    
    return
    
def examples_create_tensor():

    # 텐서를 만들어보자.
    print("**********Various ways to create Tensor************\n")

    # 파이썬 리스트에서 초기화
    tensor_from_list = torch.FloatTensor([[1,2,3],[-4,-5,-6]])
    print("tensor_from_list : ",tensor_from_list)

    # 비어있는 텐서 1
    zero_tensor = torch.zeros(2,3)
    print("zero_tensor : ",zero_tensor)

    # 비어있는 텐서 2
    empty_tensor = torch.IntTensor(2,3).zero_()
    print("empty tensor: ", empty_tensor)

    # 끝에 _(under score)가 붙은 method는 텐서 자체를 변화(mutate)시킨다.
    tensor_from_list.abs_()
    print("Apply .abs_(): ", tensor_from_list)

    # 초기값이 주어지지 않은 텐서는 임의로 초기화된다.
    uninitialized_tensor = torch.Tensor(2,3)
    print("uninitialized_tensor : ",uninitialized_tensor)

    # [0,1) uniform distribution에서 초기화
    random_tensor = torch.rand(2,3)
    print("random_tensor : ",random_tensor)

    # N(0,1) Normal distribution에서 초기화
    normal_tensor = torch.randn(2,3)
    print("normal_tensor : ",normal_tensor)

    # ndarray에서 텐서 생성
    ndarr = np.array([[1,2,3],[6,5,4]])
    from_numpy_tensor = torch.from_numpy(ndarr)
    print("from_numpy_tensor : ", from_numpy_tensor)

    # Tensor에서 ndarray 생성
    from_tensor_ndarray = from_numpy_tensor.numpy()
    print("from_tensor_ndarray : ", from_tensor_ndarray)
    
    return
    
def examples_manipulate_tensor():
    
    # numpy의 ndarray을 변화시키는 여러 함수가 정의되어 있듯, Pytorch 에서도 텐서를 변화시키는 다양한 함수가 정의되어 있습니다.
    # 텐서를 여러가지 방식으로 변화시키는 방식을 살펴봅시다.
    
    print("**********Various ways to manipulate Tensor************\n")
    
    X = torch.randn(3,5)
    print("Original : ",X)

    # Concatenation
    concat_tensor_0 = torch.cat((X,X,X),0)
    print("Concat through axis 0 :", concat_tensor_0)
    concat_tensor_1 = torch.cat((X,X,X),1)
    print("Concat through axis 1 : ", concat_tensor_1)

    # Chunking
    chunk_tensor = torch.chunk(X,3,dim=0)
    print("chunk_tensor : ", chunk_tensor)

    # Non-zero
    eye_tensor = torch.eye(3,3)
    nonzero_index = torch.nonzero(eye_tensor)
    print("nonzero_index : ", nonzero_index)

    # Transpose
    trans_tensor = torch.t(X)
    print("trans_tensor", trans_tensor)
    
    return

def examples_operate_tensor():

    # 텐서에 여러가지 연산을 적용해보자.
    print("**********Various ways to apply operation on Tensor************\n")
    
    A = torch.randn(2,2)
    B = torch.randn(2,2)

    print("Original A : ", A)
    print("Original B : ", B)

    # element-wise tensor addition
    added_tensor = torch.add(A,B)
    # or,
    added_tensor_2 = A+B
    print("Added tensor : ",added_tensor)

    # Clamping tensor
    clamp_tensor = torch.clamp(A, min=-0.5, max=0.5)
    print("clamp_tensor : ", clamp_tensor)

    # Divide
    divide_by_const_tensor = torch.div(A,2)
    divide_by_tensor = torch.div(A,B)
    # or,
    devied_by_tensor_2 = A/B
    print("divide_by_const_tensor : ",divide_by_const_tensor)
    print("divide_by_tensor : ",divide_by_tensor)

    # Element-wise multiplication
    mul_by_const_tensor = torch.mul(A,10)
    mul_by_tensor = torch.mul(A,B)
    # or,
    mul_by_tensor = A*B
    print("mul_by_const_tensor : ",mul_by_const_tensor)
    print("mul_by_tensor : ",mul_by_tensor)

    # Matrix multiplication
    matrix_mul_tensor = torch.mm(A,B)
    print("matrix_multiplication : ", matrix_mul_tensor)

    # Sigmoid
    sigmoid_tensor = torch.sigmoid(A)
    print("sigmoid_tensor : ",sigmoid_tensor)

    # Summation
    sum_tensor = torch.sum(A)
    print("sum_tensor : ",sum_tensor)

    # Mean, standard diviation
    print("Mean : ",torch.mean(A), "std :", torch.std(A))

    return
    
if __name__ == "__main__":
    main()
