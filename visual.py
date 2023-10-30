from core_1d import rho, sigma, K, L, c
from core_1d import lam, u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from net_1d import net


fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,100)

line1, = ax.plot([],[],color='orange',lw=2)
line2, = ax.plot([],[],color='blue',lw=2)

# def init():
#     line1.set_data([],[])
#     line2.set_data([],[])
#     return line1,line2



def update(frame):
    y2 = torch.tensor([])
    for val in range(161):
        torch.cat((y2,net(torch.tensor([frame*1.]))))

    x = torch.linspace(0,L,161)
    y1 = u(x,frame)
    # y2 = net(torch.tensor([frame * 1.]))
    y1 = y1.detach().numpy()
    y2 = y2.detach().numpy()
    line1.set_data(x,y1)
    line2.set_data(x,y2)
    return line1,line2



ani = FuncAnimation(fig, update, frames = 800, blit=True, interval=1)
plt.show()