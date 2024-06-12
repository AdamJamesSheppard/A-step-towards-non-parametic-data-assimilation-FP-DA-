import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dolfin import *
from mpi4py import MPI
import os 

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Process 0 is running")
else:
    print(f"Process {rank} is running")

BOUNDARY = 5

# Set up mesh
ncells_each_direction = 40
mesh = BoxMesh(MPI.COMM_SELF,
    Point(-BOUNDARY, -BOUNDARY, -BOUNDARY), 
    Point( BOUNDARY,  BOUNDARY,  BOUNDARY), 
    ncells_each_direction, 
    ncells_each_direction, 
    ncells_each_direction
)

# Define periodic boundary conditions
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(near(x[0], -BOUNDARY) or near(x[1], -BOUNDARY) or near(x[2], -BOUNDARY))

    def map(self, x, y):
        y[0] = x[0] - BOUNDARY if near(x[0], BOUNDARY) else x[0]
        y[1] = x[1] - BOUNDARY if near(x[1], BOUNDARY) else x[1]
        y[2] = x[2] - BOUNDARY if near(x[2], BOUNDARY) else x[2]

# Apply periodic boundary condition to the function space
V = FunctionSpace(mesh, 'P', 1, constrained_domain=PeriodicBoundary())

# Define the Lorenz system as the drift vector
class LorenzSystem(UserExpression):
    def eval(self, values, x):
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        values[0] = sigma * (x[1] - x[0])
        values[1] = x[0] * (rho - x[2]) - x[1]
        values[2] = x[0] * x[1] - beta * x[2]

    def value_shape(self):
        return (3,)

# Define the diffusion matrix
def diffusion():
    dw1 = np.random.normal(1, 2)
    dw2 = np.random.normal(1, 2)
    dw3 = np.random.uniform(1, 2)
    return [dw1, dw2, dw3]

Intensity = 1
dw = diffusion()
C = Constant(
    (
        (Intensity * dw[0] * dw[0], Intensity * dw[0] * dw[1], Intensity * dw[0] * dw[2]),
        (Intensity * dw[1] * dw[0], Intensity * dw[1] * dw[1], Intensity * dw[1] * dw[2]),
        (Intensity * dw[2] * dw[0], Intensity * dw[2] * dw[1], Intensity * dw[2] * dw[2])
    )
)

# Drift part of the stochastic differential equation system
FX = interpolate(LorenzSystem(degree=2), VectorFunctionSpace(mesh, 'P', 1))

# Define initial condition function
u0_function = Expression('(1.0 / (2.0 * pi * sigma_x * sigma_y * sigma_z)) * exp(-((pow(x[0] - x0, 2) / (2 * pow(sigma_x, 2))) + (pow(x[1] - y0, 2) / (2 * pow(sigma_y, 2))) + (pow(x[2] - z0, 2) / (2 * pow(sigma_z, 2)))))',
    degree=2,
    x0=0.0, 
    y0=0.0, 
    z0=0.0,  # Center coordinates
    sigma_x=1.0, 
    sigma_y=2.0, 
    sigma_z=3.0
)

# Define trial and test function and solution at previous time-step
u = TrialFunction(V)
v = TestFunction(V)
u0 = Function(V)
u0.interpolate(u0_function)

# Define steady part of the equation
def operator(u, v):
    return dot(C * grad(u), grad(v)) * dx - u * inner(FX, grad(v)) * dx

# Time-stepping parameters
T = 1
dt = 0.01
theta = Constant(1)  # Crank-Nicolson scheme for 0.5
simulation_time = dt

# Prepare solution function and solver
u = Function(V)
problem = LinearVariationalProblem(lhs((1.0/dt) * dot(u - u0, v) * dx + theta * operator(u, v) + (1.0 - theta) * operator(u0, v)),
                                   rhs((1.0/dt) * dot(u - u0, v) * dx + theta * operator(u, v) + (1.0 - theta) * operator(u0, v)),
                                   u)
solver = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "lu"
solver.parameters["preconditioner"] = "ilu"

# Main time-stepping loop
while simulation_time <= T:
    solver.solve()
    u0.assign(u)
    simulation_time += dt

# Plot the solution and save to the specified directory
plot_directory = 'fenics_run_plots_main/tests/test7_reference_solution'
os.makedirs(plot_directory, exist_ok=True)
plt.figure(figsize=(10, 6))
plot(u, title="Final Distribution")
plt.xlabel('X Coordinate')
plt.ylabel('Function Value')
plt.savefig(os.path.join(plot_directory, 'final_distribution.png'))
plt.show()
