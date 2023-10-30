import torch

rho = 8.92
sigma = .092
K = .95
L = 80
c = (K/(sigma*rho))**.5

def lam(n):
    return c * n * torch.pi / L

def u(x,t):
    return 100 * torch.e**(-lam(1)**2*t) * torch.sin((torch.pi/L)*x)



data_range = 5
num_heat_points = 161

x = torch.linspace(0,80,num_heat_points) #change back to 161 later

train_input = torch.tensor([])
train_output = torch.tensor([])


for t in range(data_range):
    for x_pos in x:
        train_input = torch.cat((train_input,torch.tensor([[x_pos,t]])))
                                
        #train_output = torch.cat((train_output,u(x,t).view(1,num_heat_points)))
        train_output = torch.cat((train_output,torch.tensor([[u(x_pos,t)]])))


print(train_output.shape)

print('input',train_input[:4])
print('output',train_output[:4])

time = torch.tensor([])
position = torch.tensor([])

for val in train_input:
    time = torch.cat((time,torch.tensor([val[1]])))
    position = torch.cat((position,torch.tensor([val[0]])))

print('time',time[:4])
print('position',position[0:4])



