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



train_til_time = 30
num_heat_points = 161

x = torch.linspace(0,80,num_heat_points) #change back to 161 later


train_output = torch.tensor([])
train_input = torch.tensor([])
final_input = torch.tensor([])
for t in range(train_til_time):
    

    train_input = torch.cat((torch.tensor([t]),x),dim=0)
    train_input = train_input.view(1,num_heat_points + 1)
    final_input = torch.cat((final_input,train_input),dim=0)


    train_output = torch.cat((train_output,u(x,t).view(1,num_heat_points)))

print(train_output.shape)



