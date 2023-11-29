import torch

rho = 8.92
sigma = .092
K = .95
L = 80
c = (K/(sigma*rho))**.5


def lam(n):
    return c * n * torch.pi / L

# u() is the analytical solution to our 1d heat equation problem
def u(x,t):
    return 100 * torch.e**(-lam(1)**2*t) * torch.sin((torch.pi/L)*x)

# how many seconds the data is trained for
data_range = 500

# how many points along the 1d bar are sampled for training
num_heat_points = 15

#tensor of all the points sampled for heat training 
x = torch.linspace(0,80,num_heat_points)

# initialiazed 1d tensors for building our input and output datasets 
train_input = torch.tensor([])
train_output = torch.tensor([])

for t in range(data_range):
    for x_pos in x:

        train_input = torch.cat((train_input,torch.tensor([[x_pos,t]])))            
        train_output = torch.cat((train_output,torch.tensor([[u(x_pos,t)]]))).requires_grad_(True)


print(train_input.shape)
train_input = torch.reshape(train_input,(data_range,num_heat_points,2))


time = torch.tensor([])
position = torch.tensor([[]])




