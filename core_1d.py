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



train_til_time = 5
num_heat_points = 161

x = torch.linspace(0,80,num_heat_points) #change back to 161 later


output = torch.tensor([])
t_train = torch.tensor([])

for t in range(train_til_time):
    t_train = torch.cat((t_train, torch.tensor([t])),dim=0)
    out = torch.tensor(u(x,t))
    out = out.view(1,num_heat_points)
    output = torch.cat((output,out),dim=0)

print(t_train)

print(output)



