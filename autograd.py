import torch

x = torch.rand(3, requires_grad=True)

y = x + 2
print(y) # grad_fn=<AddBackward0>
z = y*y*2 # grad_fn=<MulBackward0>
z = z.mean() # grad_fn=<MeanBackward0>

z.backward() # dz/dx - ONLY for scalar outputs
x_grad = x.grad # this now exists

x = torch.rand(3, requires_grad=True)
# Prevent gradient tracking
x.requires_grad_(False)
x.requires_grad_(True)
y = x.detach() # same values, gradient not required
with torch.no_grad():
    y = x + 2
    
# weights - training
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    
    model_output.backward() # now we have the gradient
    
    print(weights.grad)
    
# refresh the gradient to zero before each re-calculation
for epoch in range(3):
    model_output = (weights*3).sum() # so the gradient is clearly a vector with 3 as all of its elements
    
    model_output.backward() # now we have the gradient
    
    print(weights.grad)
    
    weights.grad.zero_() # in-place modification to 