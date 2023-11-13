import torch
import torch.nn as nn
import core_1d
from core_1d import data_range, num_heat_points
import matplotlib.pyplot as plt

alpha = core_1d.K / (core_1d.rho*core_1d.sigma)

class Net_1d(nn.Module):
    'build mlp for ld heat solution'

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN):
        super(Net_1d,self).__init__()
        self.act = nn.GELU()
        self.l_1 = nn.Linear(N_INPUT,N_HIDDEN)
        self.l_2 = nn.Linear(N_HIDDEN,N_HIDDEN)
        self.l_3 = nn.Linear(N_HIDDEN,N_HIDDEN)
        self.l_4 = nn.Linear(N_HIDDEN,N_OUTPUT)
        

    def forward(self,x):
        out = self.l_1(x)
        out = self.act(out)
        out = self.l_2(out)
        out = self.act(out)
        out = self.l_3(out)
        out = self.act(out)
        out = self.l_4(out)

        return out
    
#physics training set
p_train_in = core_1d.train_input




net = Net_1d(2,1,48)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=.001)

core_1d.train_input = torch.reshape(core_1d.train_input,(data_range*num_heat_points,2)).requires_grad_(True)



for _ in range(500):


    outputs = net(core_1d.train_input)

    deriv = torch.autograd.grad(outputs.sum(),core_1d.train_input,create_graph=True,allow_unused=True)[0]
    deriv_2_x = torch.autograd.grad(deriv[:,0].sum(),core_1d.train_input,create_graph=True,allow_unused=True)[0]


    loss = loss_func(outputs,core_1d.train_output)# + phys_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


deriv = torch.reshape(deriv,(data_range,num_heat_points,2))
x_deriv = deriv[:,:,0]
x_deriv = x_deriv.reshape(data_range,num_heat_points,1)

true_deriv = torch.autograd.grad(core_1d.u(core_1d.train_input[:,0],core_1d.train_input[:,1]).sum(), core_1d.train_input,create_graph=True,allow_unused=True)[0]
true_deriv2 = torch.autograd.grad(true_deriv[:,0].sum(), core_1d.train_input,create_graph=True,allow_unused=True)[0]

true_deriv2 = true_deriv2[:,0].reshape(data_range,num_heat_points,1)
print('truederiv2', true_deriv2.shape)
true_deriv = torch.reshape(true_deriv,(data_range,num_heat_points,2))

true_x_deriv = true_deriv[:,:,0]
true_x_deriv = true_x_deriv.reshape(data_range,num_heat_points,1)

true_t_deriv = true_deriv[:,:,1]
true_t_deriv = true_t_deriv.reshape(data_range,num_heat_points,1)

# test differential equation fit
diffeq = alpha * deriv_2_x[:,0].reshape(data_range*num_heat_points,1) - x_deriv.reshape(data_range*num_heat_points,1)
print('diffeq',diffeq.shape)
print(diffeq.abs().sum())

#true diffeq fit
#true_diff = alpha * true_deriv2.reshape(16200,1) - true_deriv2.reshape(16200,1)



# model accuracy
model_output = net(core_1d.train_input).reshape(data_range*num_heat_points)
real_output = core_1d.u(core_1d.train_input.unbind(dim=1)[0],core_1d.train_input.unbind(dim=1)[1])
accuracy = 1 - ((model_output-real_output).sum()/real_output.sum()).abs()
print('accuracy',accuracy)





#reshape for visualization
core_1d.train_input = torch.reshape(core_1d.train_input,(data_range,num_heat_points,2))



# plt.plot(torch.linspace(0,80,81).detach().numpy(),net(core_1d.train_input[:81]).detach().numpy())
# plt.show()