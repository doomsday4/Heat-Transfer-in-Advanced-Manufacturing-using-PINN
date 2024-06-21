import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import RegularGridInterpolator

# Load the data
X_star = np.loadtxt('predict_xT.txt')
tem_pred = np.loadtxt('predict_tem.txt')

# Load the exact temperature data from the original dataset
data = scipy.io.loadmat('1D/dat/thermal_fine.mat')
x = data['x'].flatten()
t = data['tt'].flatten()
Exact = data['Tem']
Exact_tem = np.real(Exact)

# Reshape the exact temperature data to match the grid
Exact_tem_reshaped = Exact_tem.T  # Transpose for proper alignment

# Create an interpolator for the exact temperature data
interpolator = RegularGridInterpolator((t, x), Exact_tem_reshaped)

# Extract the spatial and temporal coordinates from the prediction data
x_star = X_star[:, 0]
t_star = X_star[:, 1]

# Determine the number of unique points in x and t
num_x_points = len(np.unique(x_star))
num_t_points = len(np.unique(t_star))

# Reshape the prediction data for plotting
X = x_star.reshape((num_t_points, num_x_points))
T = t_star.reshape((num_t_points, num_x_points))
Tem_pred = tem_pred.reshape((num_t_points, num_x_points))

# Interpolate the exact temperature data at the prediction points
Exact_tem_resampled = interpolator((t_star, x_star)).reshape((num_t_points, num_x_points))

# Plot predicted and exact temperature distributions side by side
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.contourf(X, T, Tem_pred, levels=50, cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted Temperature Distribution')

plt.subplot(1, 2, 2)
plt.contourf(X, T, Exact_tem_resampled, levels=50, cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact Temperature Distribution')

plt.tight_layout()
plt.show()