import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from dolfin import *        
import fenics 
from mpi4py import MPI
from scipy.interpolate import LinearNDInterpolator, interp2d

###### CONSTANTS ######
### CURRENT SET-UP USES AROUND 18-20gb OF MEMORY (RAM) ###

BOUNDARY_X = 3.345      # +- also initial condition 
BOUNDARY_Y = 3.2        # +- also initial condition 
BOUNDARY_Z = 1          # +- also initial condition 

# BOUNDARIES CREATE THE COMPUTATIONAL DOMAIN OR INTERVAL WITH INITIAL AND FINAL CONDITION (boundary value inputs)
np.random.seed(42)
ncells_each_direction = 50
Intensity = 10
# Time-stepping parameters
T = 50  # 5.0
dt = 0.001       
theta = Constant(1)  # Crank-Nicolson scheme for 0.5
plot_directory = 'fenics_run_plots_main/tests/test13'
initial_dist = '/media/adam/Unix_Partition2/docker_neural_operator/docker/Code Base Fourier NO/lorenz63-fokker-planck-main/pypde/Lorenz Python Files/lorenz_hist.csv'
import os

def initialise_MPI_environment():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("Process 0 is running")
    else:
        print(f"Process {rank} is running")

def set_up_mesh():
    mesh = BoxMesh(
               MPI.COMM_WORLD,
               Point(-BOUNDARY_X, -BOUNDARY_Y, -BOUNDARY_Z),
               Point(BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z),
               ncells_each_direction,
               ncells_each_direction,
               ncells_each_direction
               )
    #plot(mesh)
    return mesh

class LorenzSystem(UserExpression):
    def eval(self, values, x):
        sigma = 3.765432        # PARAMETERS 
        rho = 17.654323456      # PARAMETERS     
        beta = 8/3              # PARAMETERS
        #print(x)
        values[0] = sigma * (x[1] - x[0])
        values[1] = x[0] * (rho - x[2]) - x[1]
        values[2] = x[0] * x[1] - beta * x[2]

    def value_shape(self):
        return (3,)

class StateSpaceDistribution(UserExpression):
    def __init__(self, f, **kwargs):
        self.LorenzSystem = f  # interpolant
        UserExpression.__init__(self, **kwargs)
        
    def eval(self, values, x):
        x_ = np.array([*x])
        #print(x_)
        result = self.LorenzSystem(x_[0], x_[1])
        if np.isnan(result): # delta function
            values[:] = 0  # Set value to zero if NaN
        else:
            values[:] = result

def diffusion():
    dw1 = np.random.normal(1, 2)
    dw2 = np.random.normal(1, 2)
    dw3 = np.random.uniform(1, 2)
    return [dw1, dw2, dw3]

def define_matrices(dw):
    C = Constant(
        (
            (Intensity * dw[0] * dw[0], Intensity * dw[0] * dw[1], Intensity * dw[0] * dw[2]),
            (Intensity * dw[1] * dw[0], Intensity * dw[1] * dw[1], Intensity * dw[1] * dw[2]),
            (Intensity * dw[2] * dw[0], Intensity * dw[2] * dw[1], Intensity * dw[2] * dw[2])
        )
    )
    C_CONTROL_INHOMOGENOUS = Constant(
        (
            (1.0, 0.2, 0.1),
            (0.1, 1.0, 0.3),
            (0.2, 0.5, 1.0)
        )
    )

    CONST_DIFFUSION = Constant(
        (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        )

    )

    return C, C_CONTROL_INHOMOGENOUS, CONST_DIFFUSION

def get_intiial_density(mesh): # TEST FOR DIFFUSION OF NORMAL DISTRIBUTION UNDER LORENZ DRIFT AND DIFFUSION MATRIX
    # class for nascent delta distribution as starting point for FPT
    u0_function = Expression('(1.0 / (2.0 * pi * sigma_x * sigma_y * sigma_z)) * exp(-((pow(x[0] - x0, 2) / (2 * pow(sigma_x, 2))) + (pow(x[1] - y0, 2) / (2 * pow(sigma_y, 2))) + (pow(x[2] - z0, 2) / (2 * pow(sigma_z, 2)))))',
                            degree=3,
                            x0=0.0,
                            y0=0.0,
                            z0=0.0,  # Center coordinates
                            sigma_x=1,
                            sigma_y=1,
                            sigma_z=1,  # Standard deviations along each axis
                            domain=mesh)
    return u0_function

def get_operator(u, v, C_CONTROL, FX): 
    return (dot(C_CONTROL * grad(u), grad(v))) * dx - u * inner(FX, grad(v)) * dx

def get_fokker_planck(u, u0, v, operator):
    return (1.0 / dt) * dot(u - u0, v) * dx + theta * operator
    
def run_loop(mesh_, u_sol, u0, solver, plot_directory):
    simulation_time = 0
    counter = 0
    progress_bar = tqdm(total=int(T / dt), desc="Solving Fokker-Planck equation")

    while simulation_time <= T:
        counter += 1

        # Solve the problem
        solver.solve()
        U_local = u0.compute_vertex_values(mesh_)
        # produce solution data from all processes
        u0.assign(u_sol)
        # Gather solution data from other processors to the root process (rank 0)
        U_all = MPI.COMM_WORLD.gather(U_local, root=0)
        # Gather vertex coordinates on process 0
        X = mesh_.coordinates()[:, 0]
        Y = mesh_.coordinates()[:, 1]
        
        # Gather coordinates from all processors
        X_all = MPI.COMM_WORLD.gather(X, root=0)
        Y_all = MPI.COMM_WORLD.gather(Y, root=0)

        # Process 0 has all the gathered data
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(plot_directory + "/data", exist_ok=True)
            data_path = f"{plot_directory}/pdf_{counter - 1}.npy"
            # Combine data from all processes into a single array
            combined_U = np.concatenate(U_all)
            combined_X = np.concatenate(X_all)
            combined_Y = np.concatenate(Y_all)
            data_array = np.stack((combined_X, combined_Y, combined_U), axis = 0)
            np.save(data_path, data_array)
            # Update progress bar
            progress_bar.update(1)
            # Perform garbage collection
            gc.collect()
        
        simulation_time += dt

    progress_bar.close()

def get_data(file_path, mesh_):
    data = np.genfromtxt(file_path, delimiter=',')
    #print(data)
    values, x, y = data[:,0], data[:,1], data[:,2]

    # Create arrays with additional elements
    values = np.concatenate([values, np.zeros(len(mesh_.coordinates()[:, 0]) - len(values))])
    x = np.concatenate([x, np.zeros(len(mesh_.coordinates()[:, 1]) - len(x))])
    y = np.concatenate([y, np.zeros(len(mesh_.coordinates()[:, 2]) - len(y))])
    interpolant = LinearNDInterpolator(list(zip(x, y)), values)
    Z = interpolant(x, y)
    #print(Z.shape, len(mesh_.coordinates()[:, 0]))
    #print(x.shape)
    #print(y.shape)

    #plt.plot(Z)
    #plt.plot(Z[:len(mesh_.coordinates()[:, 0])])
    #plt.show()
    #plt.scatter(x, y, Z, c = Z)
    #plt.show()
    print(values.shape)
    return interpolant

def boundary(x, on_boundary):
    tol = 1e-3  # Adjust the tolerance as needed
    return (
        on_boundary and 
        near(x[0], BOUNDARY_X, tol) and near(x[1], BOUNDARY_Y, tol) and near(x[2], BOUNDARY_Z, tol) and  # One set of conditions
        near(x[0], -BOUNDARY_X, tol) and near(x[1], -BOUNDARY_Y, tol) and near(x[2], 0, tol)  # Another set of conditions
    )


def main():
    
    initialise_MPI_environment();                               print('initialised message passing')
    mesh = set_up_mesh();                                       print(f'set up mesh: Size {len(mesh.coordinates()[:, 0])}')
    dw = diffusion();                                           print('initialised diffusion vector')
    C, C_CONTROL,   CONST_DIFFUSION = define_matrices(dw);         print('defined matrices')
    u0_function = get_intiial_density(mesh)
    
    # Create periodic boundary condition                        # can reduce computational complexity IF NEEDED
    #pbc = PeriodicBoundary();                                  print('defined boundary conditions: NONE')
    # Apply periodic boundary condition to the function space
    
    ### IF PERIODIC BOUNDARIES ARE EMPLOYED USE: FunctionSpace(mesh, 'P', degree = 2, constrained_boundary = pbc) ###
    
    V = FunctionSpace(mesh, 'CG', 1);                             print('defined function space')
    W = VectorFunctionSpace(mesh, 'CG', 1);                      print('defined vector function space')
    FX = interpolate(LorenzSystem(degree=2), W);                print('interpoldated lorenz system')
    
    #MPI.COMM_WORLD.Barrier()
    
    u = TrialFunction(V);                                       print('defined trial function')
    v = TestFunction(V);                                        print('defined test function')
    
    #u0 = Function(V)
    
    u_D = Constant(0)  # Homogeneous Dirichlet condition
    bc = DirichletBC(V, u_D, boundary);                         print('set boundary condition')
    u0 = interpolate(StateSpaceDistribution(get_data(initial_dist, mesh), element = V.ufl_element(), degree = 2), V)
    #u0 = interpolate(u0_function, V)
    #MPI.COMM_WORLD.Barrier()
    # REPLACE WITH C_CONTROL FOR REPRODUCIBILITY
    
    operator = get_operator(u, v, CONST_DIFFUSION, FX);               print('calculated steady state operator')
    F = get_fokker_planck(u, u0, v, operator);                  print('initialised fokker planck equation')
    
    u = Function(V);                                            print('set up solution space')
    problem = LinearVariationalProblem(lhs(F), rhs(F), u, bc);      print('set up linear variational problem')
    solver = LinearVariationalSolver(problem)
    
    solver.parameters["linear_solver"] = "gmres"                # matrix system too big for efficient direct/exact solving

    solver.parameters["preconditioner"] = "ilu"                 # for parallel computation
    u.interpolate(u0);                                          print('set initial solution to initial PDF on process: 0')
    
    #plt.scatter(mesh.coordinates()[:, 0], mesh.coordinates()[:, 1], u.compute_vertex_values(), c = u.compute_vertex_values())
    #plt.show()
    
    MPI.COMM_WORLD.Barrier()
    
    run_loop(mesh, u, u0, solver, plot_directory);              print('solving fokker planck')

class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        # Return True if on left or bottom boundary AND NOT on lower-left corner
        return bool(
            (
                    near(x[0], -BOUNDARY) or near(x[1], -BOUNDARY) or near(x[2], -BOUNDARY)
            )
            and
            (
                    not
                    (
                            near(x[0], -BOUNDARY) and near(x[1], -BOUNDARY)
                    )
                    and
                    (
                            not
                            (
                                    near(x[1], -BOUNDARY) and near(x[2], -BOUNDARY)
                            )
                    )
                    and
                    (
                            not
                            (
                                    near(x[2], -BOUNDARY) and near(x[0], -BOUNDARY)
                            )
                    )
                    and
                    (
                            not
                            (
                                    near(x[0], BOUNDARY) and near(x[1], BOUNDARY)
                            )
                    )
                    and
                    (
                            not
                            (
                                    near(x[1], BOUNDARY) and near(x[2], BOUNDARY)
                            )
                    )
                    and
                    (
                            not
                            (
                                    near(x[2], BOUNDARY) and near(x[0], BOUNDARY)
                            )
                    )
            )
        )

    def map(self, x, y):
        if near(x[0], BOUNDARY) and near(x[1], BOUNDARY) and near(x[2], BOUNDARY):
            y[0] = x[0] - BOUNDARY
            y[1] = x[1] - BOUNDARY
            y[2] = x[2] - BOUNDARY
        elif near(x[0], BOUNDARY) and near(x[1], BOUNDARY):
            y[0] = x[0] - BOUNDARY
            y[1] = x[1] - BOUNDARY
            y[2] = x[2]
        elif near(x[1], BOUNDARY) and near(x[2], BOUNDARY):
            y[0] = x[0]
            y[1] = x[1] - BOUNDARY
            y[2] = x[2] - BOUNDARY
        elif near(x[2], BOUNDARY) and near(x[0], BOUNDARY):
            y[0] = x[0] - BOUNDARY
            y[1] = x[1]
            y[2] = x[2] - BOUNDARY
        elif near(x[0], BOUNDARY):
            y[0] = x[0] - BOUNDARY
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], BOUNDARY):
            y[0] = x[0]
            y[1] = x[1] - BOUNDARY
            y[2] = x[2]
        elif near(x[2], BOUNDARY):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - BOUNDARY
        else:
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2]


if __name__ == "__main__":
    main()
    fenics.list_timings
