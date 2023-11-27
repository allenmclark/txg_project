import torch
import torch.nn as nn
import core_1d
from core_1d import data_range, num_heat_points
import matplotlib.pyplot as plt

alpha = core_1d.K / (core_1d.rho*core_1d.sigma)
print(alpha)

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




net = Net_1d(2,1,128)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=.001)

core_1d.train_input = torch.reshape(core_1d.train_input,(data_range*num_heat_points,2)).requires_grad_(True)



for _ in range(1000):


    outputs = net(core_1d.train_input)

    deriv = torch.autograd.grad(outputs.sum(),core_1d.train_input,create_graph=True,allow_unused=True)[0]
    deriv_2 = torch.autograd.grad(deriv[:,0].sum(),core_1d.train_input,create_graph=True,allow_unused=True)[0]


    physics_loss = ((alpha * deriv_2[:,0] - deriv[:,1])**2).sum()
    print('physics loss',physics_loss)


    # rename training loss
    training_loss = loss_func(outputs,core_1d.train_output)
    print('training_loss',training_loss)
    
    loss = training_loss + physics_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


deriv = torch.reshape(deriv,(data_range,num_heat_points,2))

x_deriv = deriv[:,:,0]
x_deriv = x_deriv.reshape(data_range,num_heat_points,1)

t_deriv = deriv[:,:,1]
t_deriv = t_deriv.reshape(data_range,num_heat_points,1)



true_deriv = torch.autograd.grad(core_1d.u(core_1d.train_input[:,0],core_1d.train_input[:,1]).sum(), core_1d.train_input,create_graph=True,allow_unused=True)[0]
true_deriv2 = torch.autograd.grad(true_deriv[:,0].sum(), core_1d.train_input,create_graph=True,allow_unused=True)[0]

true_deriv2 = true_deriv2[:,0].reshape(data_range,num_heat_points,1)
true_deriv = torch.reshape(true_deriv,(data_range,num_heat_points,2))

true_x_deriv = true_deriv[:,:,0]
true_x_deriv = true_x_deriv.reshape(data_range,num_heat_points,1)

true_t_deriv = true_deriv[:,:,1]
true_t_deriv = true_t_deriv.reshape(data_range,num_heat_points,1)

# test differential equation fit
diffeq = alpha * deriv_2[:,0].reshape(data_range*num_heat_points,1) - t_deriv.reshape(data_range*num_heat_points,1)
print('diff sum',diffeq.sum())

#true diffeq fit
#true_diff = alpha * true_deriv2.reshape(16200,1) - true_deriv2.reshape(16200,1)



# model accuracy
model_output = net(core_1d.train_input).reshape(data_range*num_heat_points)
real_output = core_1d.u(core_1d.train_input.unbind(dim=1)[0],core_1d.train_input.unbind(dim=1)[1])
accuracy = 1 - ((model_output-real_output).sum()/real_output.sum()).abs()
print(accuracy)

print('model diffeq test', )






#reshape for visualization
core_1d.train_input = torch.reshape(core_1d.train_input,(data_range,num_heat_points,2))

