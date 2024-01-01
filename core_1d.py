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

# data_range = time points and num_heat_points = heat points on bar
time_range = 10
num_heat_points = 20

# same as above but for phys loss (more points)
phys_time_range = 100
phys_num_heat_points = 41



# returns tuple of input and output
def make_data(time_range, num_heat_points, bar_length = 80):
    x = torch.linspace(0, bar_length, num_heat_points,requires_grad=True)
    train_input = torch.arange(0,time_range,1)
    train_input = train_input.repeat_interleave(num_heat_points)
    x = x.repeat(time_range)
    output = u(x,train_input).view(num_heat_points*time_range,1).requires_grad_(True)
    input = torch.stack((x,train_input),dim=1)

    return input, output
    


train_input, train_output = make_data(time_range, num_heat_points)
phys_input = make_data(phys_time_range, phys_num_heat_points)[0]

