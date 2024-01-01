from core_1d import rho, sigma, K, L, c
from core_1d import lam, u, num_heat_points, time_range, phys_num_heat_points, phys_time_range, phys_input
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from net_1d import net, x_deriv, true_x_deriv, true_deriv2
import core_1d

phys_input = torch.reshape(phys_input, (phys_time_range, phys_num_heat_points, 2))


#todo refactor 50 to some variable from training sample and add comment
sample_line = core_1d.u(torch.tensor([40]),torch.tensor(time_range)).detach()


fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(-5,100)

line1, = ax.plot([],[],color='orange',lw=2,label='analytical solution')
line2, = ax.plot([],[],color='blue',lw=2, label='neural network solution')
line3, = ax.plot([],[],color='green',lw=2, label='du/dx of network')
line4, = ax.plot([],[],color='black',lw=2,label='du/dx of analytical solution')
line5, = ax.plot([],[],color='red',lw=2,label='d2u/dx of analytical solution')
line6, = ax.plot([],[],color='purple')

ax.legend(loc='lower right')

def init():
    line1.set_data([],[])
    line2.set_data([],[])
    line3.set_data([],[])
    line4.set_data([],[])
    line5.set_data([],[])
    line6.set_data([],[])
    return line1,line2



def update(frame):


    x = torch.linspace(0,L,num_heat_points)
    y1 = u(x,frame)
    y1 = y1.detach().numpy()
    line1.set_data(x,y1)

    phys_x = torch.linspace(0,L, phys_num_heat_points)
    y2 = net(phys_input[frame])
    y2 = y2.detach().numpy()
    line2.set_data(phys_x,y2)

    # y3 = x_deriv[frame]
    # y3 = y3.detach().numpy()
    # line3.set_data(x,y3)

    # y4 = true_x_deriv[frame]
    # y4 = y4.detach().numpy()
    # line4.set_data(x,y4)

    # y5 = true_deriv2[frame]
    # y5 = y5.detach().numpy()
    # line5.set_data(x,y5)

    y6 = 50
    line6.set_data([0,80],[sample_line,sample_line])


    return line1,line2,line3,line4,line5


                                #correct frames to core_1d.data_range
ani = FuncAnimation(fig, update, frames = phys_time_range, blit=True, interval=20,repeat_delay=100)
plt.show()