import torch
import torch.nn as nn
import core_1d
from core_1d import data_range, num_heat_points, train_input, train_output
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

net = Net_1d(2,1,36)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=.001)

train_input = torch.reshape(train_input,(data_range*num_heat_points,2)).requires_grad_(True)

# bound_tensor = torch.tensor([])
# for val in range(10):
#     for num in [0,80]:
#         bound_tensor = torch.cat((bound_tensor,torch.tensor([[num,val]])))

#constant is number of seconds of training data
sample = 40 * num_heat_points
iterations = 500
data = train_input
accuracy_list = []
physloss_list = []

for _ in range(iterations):
    
    outputs = net(data)

    deriv = torch.autograd.grad(outputs.sum(), data,create_graph=True, allow_unused=True)[0]
    deriv_2 = torch.autograd.grad(deriv[:,0].sum(), data, create_graph=True, allow_unused=True)[0]

    true_deriv = torch.autograd.grad(core_1d.u(data[:,0], data[:,1]).sum(), data, create_graph=True,allow_unused=True)[0]
    true_deriv2 = torch.autograd.grad(true_deriv[:,0].sum(), data, create_graph=True, allow_unused=True)[0]


    #test with solution derivs
    physics_loss = torch.mean((alpha * deriv_2[:,0] - deriv[:,1])**2)
    physloss_list.append(physics_loss.detach())
    # boundary_loss = torch.mean(net(bound_tensor)**2)*50
    


    #rename training loss and change 150s to variables
    training_loss = (outputs[0:sample] - train_output[0:sample]).flatten()
    training_loss = torch.cat((training_loss,torch.zeros(data_range*num_heat_points-sample)),0)
    training_loss = torch.mean((outputs - train_output)**2)
    

    loss = training_loss + physics_loss * 100 # + boundary_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



    model_output = net(data).reshape(data_range*num_heat_points)
    real_output = core_1d.u(data.unbind(dim=1)[0], data.unbind(dim=1)[1])
    accuracy = 1 - ((model_output-real_output).sum()/real_output.sum()).abs()
    

    accuracy_list.append(accuracy.detach())
    
        
    if _ % 200 == 0:
        print('accuracy',accuracy)

    

print(accuracy_list)
plt.plot(accuracy_list,color='black')
plt.show()
plt.plot(physloss_list,color='red')
plt.show()





print(deriv_2[:,0] - true_deriv2[:,0])
print((deriv_2[:,0] - true_deriv2[:,0])**2)
print(((deriv_2[:,0] - true_deriv2[:,0])**2).sum())

plt.plot(deriv_2[:,0].detach() - true_deriv2[:,0].detach(),color='red')
plt.show()



deriv = torch.reshape(deriv,(data_range,num_heat_points,2))

x_deriv = deriv_2[:,0]
x_deriv = x_deriv.reshape(data_range,num_heat_points,1)

t_deriv = deriv[:,:,1]
t_deriv = t_deriv.reshape(data_range,num_heat_points,1)



true_deriv = torch.autograd.grad(core_1d.u(data[:,0],data[:,1]).sum(), data,create_graph=True,allow_unused=True)[0]
true_deriv2 = torch.autograd.grad(true_deriv[:,0].sum(), data,create_graph=True,allow_unused=True)[0]


# plt.plot((alpha * true_deriv2[:,0]).detach(),color='black') 
# plt.plot(2*true_deriv[:,1].detach(),color='red')
# plt.show()

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
model_output = net(data).reshape(data_range*num_heat_points)
real_output = core_1d.u(data.unbind(dim=1)[0],data.unbind(dim=1)[1])
accuracy = 1 - ((model_output-real_output).sum()/real_output.sum()).abs()
print('accuracy',accuracy)











#reshape for visualization
#core_1d.train_input = torch.reshape(data,(data_range,num_heat_points,2))






