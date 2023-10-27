import torch
import torch.nn as nn
import core_1d
import matplotlib.pyplot as plt


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
        out = self.l_3(out)

        return out
    
train_in = torch.linspace(0,300,300)
train_in = train_in.view(300,1)


net = Net_1d(1,161,210)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=.01)

for val in range(5000):
    outputs = net(train_in)

    loss = loss_func(outputs,core_1d.train_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


net(torch.tensor([200.])).shape

plt.plot(torch.linspace(0,80,161).detach().numpy(),net(torch.tensor([800.])).detach().numpy())
plt.show()