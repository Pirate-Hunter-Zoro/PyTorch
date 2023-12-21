import torch
import numpy as np

x = torch.ones(2,2)

x = torch.ones(2,2, dtype=torch.int)

x = torch.ones(2,2, dtype=torch.float32)

x = torch.tensor([2.5,0.1])

# vector arithmetic - element-wise
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x + y
z = torch.add(x,y) # addition
y.add_(x) # in-place modification
z = x - y
z = torch.sub(x,y)
z = torch.mul(x,y)
y.mul_(x)
x.div_(y)

# slicing
x = torch.rand(5,3)
print(x)
x[:,0] # first column, all rows
x[1,:] # second row, all columns
x[1,1].item() # if tensor with one element, get actual value

# reshaping
x = torch.rand(4,4)
print(x)
y = x.view(16)
z = x.view(-1,8) # pytorch figures out the -1 meaning 2
print(y)

# conversion from tensor to numpy array
a = torch.ones(5)
b = a.numpy() # numpy.ndarray
# BUT BE CAREFUL
a.add_(1)
# now both are changed because they share the same memory address since we're on the CPU - whatever the fuck that means

# conversion from numpy array to tensor
a = np.ones(5)
b = torch.from_numpy(a)
# SAME MEMORY ADDRESS STILL
a += 1
# now both are changed if we're on the CPU and not the GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) # creates tensor and puts on GPU
    y = torch.ones(5)
    y = y.to(device) # that's another way of doing it
    z = x + y
    # z.numpy() - returns an error because 'numpy()' can only handle CPU tensors
    z = z.to("cpu")
    
# Pytorch knows it will need to calculate this tensor's gradient elements
x = torch.ones(5, requires_grad=True)