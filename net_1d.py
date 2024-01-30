import torch
import torch.nn as nn
import core_1d
from core_1d import time_range, num_heat_points, train_input, train_output, phys_input, phys_time_range, phys_num_heat_points
import matplotlib.pyplot as plt

PINN = input('Run as a Pinn? type y for yes and n for no\n\n') == 'y'

alpha = core_1d.K / (core_1d.rho*core_1d.sigma)

class Net_1d(nn.Module):
    'build mlp for ld heat solution'

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN):
        super(Net_1d,self).__init__()
        self.act = nn.GELU()
        self.l_1 = nn.Linear(N_INPUT,N_HIDDEN)
        self.l_2 = nn.Linear(N_HIDDEN,N_HIDDEN)
        self.l_3 = nn.Linear(N_HIDDEN,N_HIDDEN)
        self.l_4 = nn.Linear(N_HIDDEN,N_HIDDEN)
        self.l_5 = nn.Linear(N_HIDDEN,N_OUTPUT)
        

    def forward(self,x):
        out = self.l_1(x)
        out = self.act(out)
        out = self.l_2(out)
        out = self.act(out)
        out = self.l_3(out)
        out = self.act(out)
        out = self.l_4(out)
        out = self.act(out)
        out = self.l_5(out)

        return out

net = Net_1d(2,1,34)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=.0001)


bound_tensor = torch.tensor([])
for val in range(10):
    for num in [0,80]:
        bound_tensor = torch.cat((bound_tensor,torch.tensor([[num,val]])))

#constant is number of seconds of training data
iterations = 25_000
data = train_input
accuracy_list = []
physloss_list = []

for _ in range(iterations):
    
    data_outputs = net(data)
    phys_outputs = net(phys_input)
    
    
    deriv = torch.autograd.grad(phys_outputs, phys_input, torch.ones_like(phys_outputs), create_graph=True, allow_unused=True)[0]
    deriv_2 = torch.autograd.grad(deriv[:,0], phys_input,torch.ones_like(deriv[:,0]), create_graph=True, allow_unused=True)[0]


    # true_deriv = torch.autograd.grad(core_1d.u(data[:,0], data[:,1]).sum(), data, create_graph=True,allow_unused=True)[0]
    # true_deriv2 = torch.autograd.grad(true_deriv[:,0].sum(), data, create_graph=True, allow_unused=True)[0]


    #test with solution derivs
    physics_loss = torch.mean((alpha * deriv_2[:,0] - deriv[:,1])**2)
    physloss_list.append(physics_loss.detach())
    boundary_loss = torch.mean(net(bound_tensor)**2)
    

    training_loss = torch.mean((data_outputs - train_output)**2)
    
    if PINN == True:
        loss = training_loss + physics_loss * 100 + boundary_loss * 350
    else:
        loss = training_loss
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()



    model_output = net(data).reshape(time_range*num_heat_points)
    real_output = core_1d.u(data.unbind(dim=1)[0], data.unbind(dim=1)[1])
    accuracy = 1 - ((model_output-real_output).sum()/real_output.sum()).abs()
    

    accuracy_list.append(accuracy.detach())
    
        
    if _ % 200 == 0:
        print('accuracy',accuracy)
    

    

plt.plot(accuracy_list,color='black')
plt.show()
plt.plot(physloss_list,color='red')
plt.show()








deriv = torch.reshape(deriv,(phys_time_range, phys_num_heat_points,2))

x_deriv = deriv_2[:,0]
x_deriv = x_deriv.reshape(phys_time_range, phys_num_heat_points,1)

t_deriv = deriv[:,:,1]
t_deriv = t_deriv.reshape(phys_time_range, phys_num_heat_points,1)



true_deriv = torch.autograd.grad(core_1d.u(data[:,0],data[:,1]).sum(), data,create_graph=True,allow_unused=True)[0]
true_deriv2 = torch.autograd.grad(true_deriv[:,0].sum(), data,create_graph=True,allow_unused=True)[0]



true_deriv2 = true_deriv2[:,0].reshape(time_range,num_heat_points,1)
true_deriv = torch.reshape(true_deriv,(time_range,num_heat_points,2))

true_x_deriv = true_deriv[:,:,0]
true_x_deriv = true_x_deriv.reshape(time_range,num_heat_points,1)

true_t_deriv = true_deriv[:,:,1]
true_t_deriv = true_t_deriv.reshape(time_range,num_heat_points,1)


# model accuracy
model_output = net(data).reshape(time_range*num_heat_points)
real_output = core_1d.u(data.unbind(dim=1)[0],data.unbind(dim=1)[1])
accuracy = 1 - ((model_output-real_output).sum()/real_output.sum()).abs()
print('accuracy',accuracy)






#reshape for visualization
#core_1d.train_input = torch.reshape(data,(data_range,num_heat_points,2))






