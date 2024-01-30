import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

print("2D heat equation solver with enhanced precision")

plate_length = 100  # Increased for finer grid
max_iter_time = 1500  # Adjusted for more iterations
alpha = 20
delta_x = 0.1  # Reduced for smaller spatial steps
delta_t = (delta_x ** 2) / (4 * alpha)  # Recalculated for stability
gamma = (alpha * delta_t) / (delta_x ** 2)

# Initialize solution with finer grid spacing
u = np.empty((max_iter_time, plate_length, plate_length))

# Initial and boundary conditions
u_initial = 0
u_top = 100.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0

# Set initial and boundary conditions
u.fill(u_initial)
u[:, (plate_length - 1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (plate_length - 1):] = u_right

def calculate(u):
    for k in range(max_iter_time - 1):
        for i in range(1, plate_length - 1):  # Iterate over all grid points
            for j in range(1, plate_length - 1):
                u[k + 1, i, j] = gamma * (
                    u[k, i + 1, j] + u[k, i - 1, j] + u[k, i, j + 1] + u[k, i, j - 1] - 4 * u[k, i, j]
                ) + u[k, i, j]
    return u



def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

# Do the calculation here
u = calculate(u)

def animate(k):
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save("less_precise_heat.gif")

print("Done!")