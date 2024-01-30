import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import cm

# plt.style.use('_mpl-gallery')

data = np.load('2d_data.npy')


data = data.reshape(1000,50*50,4)

x = data[0,:,2]
y = data[0,:,1]






import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Your data points


# Create a grid of points for interpolation
xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 50**2), np.linspace(y.min(), y.max(), 50**2))



# Create the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# Customize labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-values')
ax.set_title('Continuous Surface Plot using Interpolation')


def init():
    # Create the initial surface plot
    temps = data[0, :, 3]  # Use initial data for the first frame
    zi = griddata((x, y), temps, (xi, yi), method='linear')
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis')
    return surf,  # Return the surface object

def update(frame):
    temps = data[frame, :, 3]
    zi = griddata((x, y), temps, (xi, yi), method='linear')



    # Plot the updated surface
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis')
    return surf,  # Return the updated surface object

# Create the animation
ani = FuncAnimation(fig, update, init_func=init, frames=1000, interval=200, blit=True)
plt.show()