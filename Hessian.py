import torch
x1 = torch.tensor(1.0,requires_grad=True,device='cuda:0')
x0 = torch.tensor(1.0,requires_grad=True,device='cuda:0')
y = torch.add(x1**2,x0**2)
grad = torch.autograd.grad(y,[x0,x1],retain_graph=True,create_graph=True)

print("grad is",grad,len(grad))
print("grad0 is",grad[0])
for g_ in grad:
    hessianmatrix = torch.autograd.grad(g_,[x0,x1],retain_graph=True)
    
    print(hessianmatrix)