import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import cm


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# plt.style.use('_mpl-gallery')

data = np.load('2d_data.npy')


data = data.reshape(1000,50*50,4)

x = data[0,:,2]
y = data[0,:,1]






# Your data points


# Create the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a grid of points for interpolation
xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 50**2), np.linspace(y.min(), y.max(), 50**2))



# Customize labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-values')
ax.set_title('Continuous Surface Plot using Interpolation')

def update(frame):

    temps = data[frame,:,3]
    # Interpolate the z values onto the grid
    zi = griddata((x, y), temps, (xi, yi), method='linear')  # Choose an appropriate interpolation method



    # Plot the interpolated surface
    obj = ax.plot_surface(xi, yi, zi, cmap='viridis')

    return obj


# Show the plot
ani = FuncAnimation(fig, update,frames = 1000, interval=200)
plt.show()
