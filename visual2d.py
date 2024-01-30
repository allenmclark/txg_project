import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import cm

# plt.style.use('_mpl-gallery')

data = np.load('2d_data.npy')

data = data[500]
data = data.reshape(50*50,4)

x = data[:,2]
y = data[:,1]
t = data[:,0]
temps = data[:,3]


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D



# # Create a 3D scatter plot (more suitable for individual points)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the points
# ax.scatter(x, y, temps, c=temps, cmap='viridis')  # Use z values for color variation

# # Customize labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-values')
# ax.set_title('3D Scatter Plot with Individual Points')

# # Show the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Your data points


# Create a grid of points for interpolation
xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 50**2), np.linspace(y.min(), y.max(), 50**2))

# Interpolate the z values onto the grid
zi = griddata((x, y), temps, (xi, yi), method='linear')  # Choose an appropriate interpolation method

# Create the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the interpolated surface
ax.plot_surface(xi, yi, zi, cmap='viridis')

# Customize labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-values')
ax.set_title('Continuous Surface Plot using Interpolation')

# Show the plot
plt.show()
