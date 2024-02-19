import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

alpha = 2

num_seconds = 50

PINN = input('Run as a Pinn? type y for yes and n for no\n\n') == 'y'

# last dim in form time, y, x, temperature
train_data = np.load('2d_data.npy')
train_data = torch.from_numpy(train_data).requires_grad_(True).float()
train_data = train_data[:num_seconds * 5:5]
train_data = train_data.reshape(num_seconds*50*50,4)
input_train_data, output_train_data = train_data[:,:3], train_data[:,3]


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

net = Net_1d(3,1,124)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=.01)

# REDEFINE BOUNDARY CONDITIONS


#constant is number of seconds of training data
iterations = 200000
accuracy_list = []
physloss_list = []
phys_data = None

for _ in range(iterations):
    
    model_output = net(input_train_data)
    # phys_output = net(phys_data)
    #phys_outputs = net(phys_input)
    
 

    print(model_output.shape)
    print(output_train_data.shape)
    training_loss = torch.mean((model_output - output_train_data.reshape(num_seconds*50*50,1)).abs())
    print('training loss', training_loss)

    #test with solution derivs
    physics_loss = 0
    # if training_loss < 1:
    #     print('before')
    #     deriv = torch.autograd.grad(model_output, input_train_data, torch.ones_like(model_output), create_graph=True, allow_unused=True)[0]
    #     print('middle')
    #     deriv_2 = torch.autograd.grad(deriv, input_train_data, torch.ones_like(deriv), create_graph=True, allow_unused=True)[0]
    #     print('after')

    #     physics_loss = torch.mean((alpha * (deriv_2[:,1] + deriv_2[:,2]) - deriv[:,0]).abs())

        #print('using physics loss', physics_loss)
    



    # physloss_list.append(physics_loss.detach())
    # boundary_loss = torch.mean(net(bound_tensor)**2)


    y_50  = (100 - model_output.reshape(num_seconds,50,50,1)[:,-1,1:-1].mean()).abs()
    y_0 = model_output.reshape(num_seconds,50,50,1)[:,0,:].mean().abs()
    x_0 = model_output.reshape(num_seconds,50,50,1)[:,:,0].mean().abs()
    x_50 = model_output.reshape(num_seconds,50,50,1)[:,:,-1].mean().abs()
    boundary_loss = y_50 + x_0 + x_50
    print(boundary_loss, ' is boundary loss')

    
    if PINN == True:
        loss = training_loss + physics_loss + boundary_loss
    else:
        loss = training_loss
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if loss < .5:
        print('loss is :', loss)
        torch.save(model_output,'2d_pred_closehalf.pt',)
        
    if loss < .1:
        torch.save(model_output,'2d_pred_superclose.pt',)
        break
    
        
    

    

plt.plot(accuracy_list,color='black')
plt.show()
plt.plot(physloss_list,color='red')
plt.show()








# deriv = torch.reshape(deriv,(phys_time_range, phys_num_heat_points,2))

# x_deriv = deriv_2[:,0]
# x_deriv = x_deriv.reshape(phys_time_range, phys_num_heat_points,1)

# t_deriv = deriv[:,:,1]
# t_deriv = t_deriv.reshape(phys_time_range, phys_num_heat_points,1)



# true_deriv = torch.autograd.grad(core_1d.u(data[:,0],data[:,1]).sum(), data,create_graph=True,allow_unused=True)[0]
# true_deriv2 = torch.autograd.grad(true_deriv[:,0].sum(), data,create_graph=True,allow_unused=True)[0]



# true_deriv2 = true_deriv2[:,0].reshape(time_range,num_heat_points,1)
# true_deriv = torch.reshape(true_deriv,(time_range,num_heat_points,2))

# true_x_deriv = true_deriv[:,:,0]
# true_x_deriv = true_x_deriv.reshape(time_range,num_heat_points,1)

# true_t_deriv = true_deriv[:,:,1]
# true_t_deriv = true_t_deriv.reshape(time_range,num_heat_points,1)


# # model accuracy
# model_output = net(data).reshape(time_range*num_heat_points)
# real_output = core_1d.u(data.unbind(dim=1)[0],data.unbind(dim=1)[1])
# accuracy = 1 - ((model_output-real_output).sum()/real_output.sum()).abs()
# print('accuracy',accuracy)






#reshape for visualization
#core_1d.train_input = torch.reshape(data,(data_range,num_heat_points,2))

