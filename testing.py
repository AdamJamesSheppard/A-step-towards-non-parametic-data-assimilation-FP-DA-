import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
Lx = Ly = Lz = 10.0  # Length of the domain
nx = ny = nz = 50    # Number of spatial grid points in x, y, and z directions
nt = 100             # Number of time steps
dt = 0.01            # Time step
D = 1.0              # Diffusion coefficient
vx = vy = vz = 1.0   # Drift velocities in x, y, and z directions

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dz = Lz / (nz - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)
X, Y, Z = np.meshgrid(x, y, z)

# Initial condition (Gaussian distribution)
u0 = np.exp(-(X - Lx/2)**2 / 2) * np.exp(-(Y - Ly/2)**2 / 2) * np.exp(-(Z - Lz/2)**2 / 2)

# Plot initial condition
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c=u0.flatten(), cmap='viridis')
ax.set_title('Fokker-Planck Equation - Initial Condition')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Finite difference scheme to solve the 3D Fokker-Planck equation
u = np.copy(u0)

for n in range(nt):
    un = np.copy(u)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                u[i, j, k] = un[i, j, k] + D * dt / dx**2 * (un[i+1, j, k] - 2 * un[i, j, k] + un[i-1, j, k]) \
                                          + D * dt / dy**2 * (un[i, j+1, k] - 2 * un[i, j, k] + un[i, j-1, k]) \
                                          + D * dt / dz**2 * (un[i, j, k+1] - 2 * un[i, j, k] + un[i, j, k-1]) \
                                          - vx * dt / (2 * dx) * (un[i+1, j, k] - un[i-1, j, k]) \
                                          - vy * dt / (2 * dy) * (un[i, j+1, k] - un[i, j-1, k]) \
                                          - vz * dt / (2 * dz) * (un[i, j, k+1] - un[i, j, k-1])

# Set negative values to zero
u[u < 0]= 0

# Plot final solution
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, u.flatten(), c=u.flatten(), cmap='viridis')
ax.set_title('Fokker-Planck Equation - Final Solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
