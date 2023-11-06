import torch
import torch.nn as nn
import core_1d
import matplotlib.pyplot as plt

alpha = core_1d.K / (core_1d.rho*core_1d.sigma)

class Net_1d(nn.Module):
    'build mlp for ld heat solution'

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN):
        super(Net_1d,self).__init__()
        self.act = nn.ReLU()
        self.l_1 = nn.Linear(N_INPUT,N_HIDDEN)
        self.l_2 = nn.Linear(N_HIDDEN,N_HIDDEN)
        self.l_3 = nn.Linear(N_HIDDEN,N_OUTPUT)
        

    def forward(self,x):
        out = self.l_1(x)
        out = self.act(out)
        out = self.l_2(out)
        out = self.act(out)
        out = self.l_2(out)
        out = self.act(out)
        out = self.l_3(out)

        return out
    
#physics training set
p_train_in = core_1d.train_input




net = Net_1d(2,1,128)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=.01)

core_1d.train_input = torch.reshape(core_1d.train_input,(16200,2)).requires_grad_(True)


# for val in range(len(core_1d.train_output)):
for _ in range(1000):

    
    # yhat_phys = net(p_train_in)
    # du = torch.autograd.grad(yhat_phys,core_1d.train_input,torch.ones_like(yhat_phys),create_graph=True)[0]
    # print(du)
    # du2 = torch.autograd.grad(du,p_train_in,torch.ones_like(du),create_graph=True)[0]
    # print(du2)
    # phys_res = du - alpha * du2**2
    # phys_loss = torch.mean(phys_res**2)


    outputs = net(core_1d.train_input)

    deriv = torch.autograd.grad(outputs.sum(),core_1d.train_input,create_graph=True,allow_unused=True)
    true_deriv = torch.autograd.grad(core_1d.u(core_1d.train_input[:,0],core_1d.train_input[:,1]).sum(), core_1d.train_input,create_graph=True,allow_unused=True)

    loss = loss_func(outputs,core_1d.train_output)# + phys_loss
    #print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


deriv = torch.reshape(deriv[0],(200,81,2))
x_deriv = deriv[:,:,0]
x_deriv = x_deriv.reshape(200,81,1)

print(true_deriv)
true_deriv = torch.reshape(true_deriv[0],(200,81,2))
true_x_deriv = true_deriv[:,:,0]
true_x_deriv = true_x_deriv.reshape(200,81,1)


core_1d.train_input = torch.reshape(core_1d.train_input,(core_1d.data_range,81,2))



# plt.plot(torch.linspace(0,80,81).detach().numpy(),net(core_1d.train_input[:81]).detach().numpy())
# plt.show()