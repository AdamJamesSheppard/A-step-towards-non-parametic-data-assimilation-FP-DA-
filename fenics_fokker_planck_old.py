import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from dolfin import *
from fenics import *
from matplotlib import cm
import os
from scipy.integrate import odeint
from tqdm import tqdm  # Import tqdm library for progress bar

# model parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

T = 1  # final time
dt = 0.1  # T / num_steps # time step size
num_steps = int(T / dt)  # number of time steps
np.random.seed(42)
Intesity = 1

print("time step", dt)
# Create mesh and define function space
nx = ny = nz = 40

mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), nx, ny, nz)
n_ = FacetNormal(mesh)
V = FunctionSpace(mesh, 'P', 1)
W = VectorFunctionSpace(mesh, 'P', 1)

def white_noise():
    dw1 = np.random.normal(0,1)
    dw2 = np.random.normal(0,1)
    dw3 = np.random.uniform(0,1)
    return dw1, dw2, dw3

dw = white_noise()


# Define diffusion matrix (D)
D_matrix = Constant(
    (
        (Intesity*dw[0]*dw[0], Intesity*dw[0]*dw[1], Intesity*dw[0]*dw[2]),
        (Intesity*dw[1]*dw[0], Intesity*dw[1]*dw[1], Intesity*dw[1]*dw[2]),
        (Intesity*dw[2]*dw[0], Intesity*dw[2]*dw[1], Intesity*dw[2]*dw[2])
    )
)

velocity = Expression(
    (
        'sigma * (x[1]        - x[0])' , 
        'x[0]  * (rho - x[2]) - x[1]' , 
        'x[0]  * x[1] - beta  * x[2]'
    ),  # LORENZ
            sigma     = sigma,
            rho       = rho,
            beta      = beta,
            dw1       = white_noise()[0],
            dw2       = white_noise()[1],
            dw3       = white_noise()[2],
            Intensity = Intesity,
            degree    = 3)

vel = interpolate(velocity, W)

# Define variational problem
u   = TrialFunction(V)
v   = TestFunction(V)
u_  = Function(V)
u_n = Function(V)

# Define initial distribution
u0  = Expression(
    '(1.0 / (2.0 * pi * sigma_x * sigma_y * sigma_z)) * exp(-((pow(x[0] - x0, 2) / (2 * pow(sigma_x, 2))) + (pow(x[1] - y0, 2) / (2 * pow(sigma_y, 2))) + (pow(x[2] - z0, 2) / (2 * pow(sigma_z, 2)))))',
            degree   = 3,
            x0       = 0.0, 
            y0       = 0.0, 
            z0       = 0.0,  # Center coordinates
            sigma_x  = 0.3, 
            sigma_y  = 0.3, 
            sigma_z  = 0.3)  # Standard deviations along each axis

u_n = interpolate(u0, V)

k   = Constant(dt)

# forward finite differences
#  drift      (volume)
#  diffusion  (volume)
#  diffusion  (surface)
#  drift      (surface)
F = dot( (u - u_n)/k, v) * dx +\
    inner(grad(u), grad(v)) * dx \
     -   u * inner(vel, grad(v))  * dx \
     +  v * inner(grad(u), n_) * ds\
     -  u * v * inner(vel, n_) * ds \

a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile = File('Fokker-planck_stochastic_large_time_step_Solution/fokker-planck.pvd')

# Time-stepping
t = 0

A = assemble(a)
b = assemble(L)

plot_directory = 'Fokker-planck_stochastic_pdf_large_time_step_graphs'
os.makedirs(plot_directory, exist_ok=True)

# Plot initial distribution
plt.figure(figsize=(30, 30))
plt.plot(u_n.compute_vertex_values(mesh))
# Save plot
plt.savefig(f'{plot_directory}/initial_distribution.png')
plt.close()
print(num_steps)

# Initialize tqdm with total number of steps
progress_bar = tqdm(total=num_steps, desc="Solving Fokker-Planck equation")

for n in range(num_steps):
    t += dt

    u_n.rename("probability", "")
    vtkfile << (u_n, t)
    A = assemble(a)
    b = assemble(L)

    # Compute solution
    solve(A, u_.vector(), b)

    # Update previous solution
    u_n.assign(u_)
    pdf_values = u_.compute_vertex_values(mesh)
    
    # Create a figure with specified size
    fig = plt.figure(figsize=(30, 30))

    # Plot the normalized PDF values
    plt.plot(pdf_values, label='PDF Values')
    plt.savefig(f'{plot_directory}/pdf_1d_plot_at_t_{n}.png')
    plt.close()

    # Update progress bar
    progress_bar.update(1)

# Close tqdm progress bar
progress_bar.close()

u_n.rename("probability", "")
vtkfile << (u_n, t)
