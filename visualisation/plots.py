import numpy as np
import matplotlib.pyplot as plt

# Load the data
X_star = np.loadtxt('predict_xT.txt')
tem_pred = np.loadtxt('predict_tem.txt')
ftem_pred = np.loadtxt('predict_ftem.txt')

# Extract the spatial and temporal coordinates
x = X_star[:, 0]
t = X_star[:, 1]

# Determine the number of unique points in x and t
num_x_points = len(np.unique(x))
num_t_points = len(np.unique(t))

# Reshape the data for plotting
X = x.reshape((num_x_points, num_t_points))
T = t.reshape((num_x_points, num_t_points))
Tem = tem_pred.reshape((num_x_points, num_t_points))

# Plot temperature distribution
plt.figure()
plt.contourf(X, T, Tem, levels=50, cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted Temperature Distribution')
plt.show()

# Reshape the residuals
Res = ftem_pred.reshape((num_x_points, num_t_points))

# Plot residuals
plt.figure()
plt.contourf(X, T, Res, levels=50, cmap='coolwarm')
plt.colorbar(label='Residuals')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Residuals of the Heat Equation')
plt.show()