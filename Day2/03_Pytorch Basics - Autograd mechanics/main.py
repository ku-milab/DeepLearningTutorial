import elice_utils
import torch
from torch.autograd import Variable

def main():
    #x = torch.ones(2,2)
    #Create_Variable(x)
    #Operation_of_Variable(x)
    #example_1()
    #example_2()
    #excluding_subgraph()
    #Two_layer_NN()
    
def example_1():
    # Create a Variable that wraps a simple tensor
    V_no_grad = Variable(torch.ones(2,2))
    V = Variable(torch.ones(2,2),requires_grad=True)
    
    print("type",type(V))
    print("Default requires_grad?",V_no_grad.requires_grad) # default = False
    
    print("raw data : ",V.data) # a raw Tensor wrapped by Variable
    print("initial gradient : ",V.grad) # no gradient accumulated yet
    print("Creator? : ",V.grad_fn) # This Variable is created by User, not a autograd.Function
    
    # A Variable operation creates another Variable
    child = V + 2 # Add Constant 
    
    print("raw data : ",child.data)
    print("initial gradient : ",child.grad)
    print("Creator? : ",child.grad_fn) # You can see autograd.Function class that create this Variable
    
    return

def example_2():
    A = Variable(torch.eye(2,2),requires_grad=True)
    B = Variable(torch.ones(2,2),reqires_grad=True)
    
    inter_1 = A+3*B
    inter_2 = inter_1**2
    out = inter_2.mean()
    
    out.backward()
    
    # Check that B.grad = A.grad * 3
    print("Gradient of A : ", A.grad)
    print("Gradient of B : ", B.grad)
    
    return
    
def Create_Variable(tensor):
    # Create Variable from torch.Tensor
    Var = Variable(tensor, requires_grad=True)
    print(Var)
    return Var

def Operation_of_Variable(tensor):
    # Create Variable from torch.Tensor
    Var = Variable(tensor, requires_grad=True)
    
    # Simple addition (Change if you want to)
    K = Var + 2
    print(K)
    
    # K is a result of an operation. so it has grad_fn
    print(K.grad_fn)
    
    # another operation
    M = Var * Var * 3
    Out = M.mean()
    print(M, Out)
    print(Out.grad_fn)
    
def excluding_subgraph():
    
    # Output has requires_grad=True if one of its leaves has requires_grad=True
    A = Variable(torch.randn(2,2),requires_grad=False)
    B = Variable(torch.randn(2,2),requires_grad=True)
    C = A+B
    print("C.requires_grad : ",C.requires_grad)
    
    # If an input has flag "Volatile=True", then the output has requires_grad=False
    AA = Variable(torch.randn(2,2), Volatile=True)
    BB = Variable(torch.randn(2,2), requires_grad=True)
    CC = AA+BB
    print("CC.requires_grad : ",CC.requires_grad)
    
    return
    
def Two_layer_NN():

    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold input and outputs, and wrap them in Variables.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Variables during the backward pass.
    x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

    # Create random Tensors for weights, and wrap them in Variables.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Variables during the backward pass.
    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

    learning_rate = 1e-6
    
    for t in range(500):
        # Forward pass: compute predicted y using operations on Variables; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Compute and print loss using operations on Variables.
        # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
        # (1,); loss.data[0] is a scalar value holding the loss.
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.data[0])

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Variables with requires_grad=True.
        # After this call w1.grad and w2.grad will be Variables holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Update weights using gradient descent; w1.data and w2.data are Tensors,
        # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
        # Tensors.
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        # Manually zero the gradients 
        w1.grad.data.zero_()
        w2.grad.data.zero_()
    
    print("Prediction : ", y_pred.data)
    print("Label : ", y.data)
    return

if __name__ == "__main__":
    main()
