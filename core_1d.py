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

# how many seconds the model is trained for
data_range = 300

# how many points along the 1d bar are sampled for training
num_heat_points = 20

#tensor of all the points sampled for heat training 
x = torch.linspace(0,80,num_heat_points)

# create tensor of temps for half of bar and zeros for other half
phys_x = x[:int(num_heat_points/2)]
phys_x = torch.cat((phys_x,torch.zeros(int(num_heat_points/2))))

# initialiazed 1d tensors for building our input and output datasets 
train_input = torch.tensor([])
train_output = torch.tensor([])
phys_train_input = torch.tensor([],requires_grad=True)
phys_train_output = torch.tensor([], requires_grad=True)

for t in range(data_range):
    for x_pos in x:
        train_input = torch.cat((train_input,torch.tensor([[x_pos,t]])))            
        train_output = torch.cat((train_output,torch.tensor([[u(x_pos,t)]]))).requires_grad_(True)
    for pos in phys_x:
        phys_train_input = torch.cat((phys_train_input,torch.tensor([[pos,t]]))) 
        phys_train_output = torch.cat((phys_train_output,torch.tensor([[u(pos,t)]]))).requires_grad_(True)


train_input = torch.reshape(train_input,(data_range,num_heat_points,2))







