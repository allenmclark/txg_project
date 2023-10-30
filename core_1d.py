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

print(u(torch.tensor([30,40]),torch.tensor([[2,3]])))

# how many seconds the data is trained for
data_range = 5

# how many points along the 1d bar are sampled for training
num_heat_points = 81

#tensor of all the points sampled for heat training 
x = torch.linspace(0,80,num_heat_points)

# initialiazed 1d tensors for building our input and output datasets 
train_input = torch.tensor([])
train_output = torch.tensor([])

for t in range(data_range):
    for x_pos in x:
        train_input = torch.cat((train_input,torch.tensor([[x_pos,t]])))
                                
        #train_output = torch.cat((train_output,u(x,t).view(1,num_heat_points)))
        train_output = torch.cat((train_output,torch.tensor([[u(x_pos,t)]])))


print('input',train_input.shape)
print('output',train_output.shape)

time = torch.tensor([])
position = torch.tensor([[]])

# for val in train_input:
#     position = torch.cat((position,torch.tensor([[val[0]]])))
#     time = torch.cat((time,torch.tensor([val[1]])))


# print('time',time[:4])
# print('position',position)



