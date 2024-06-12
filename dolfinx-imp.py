import matplotlib.pyplot as plt
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh, io
import ufl

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 50
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],
                               [nx, ny], mesh.CellType.triangle, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
V = fem.FunctionSpace(domain, ("Lagrange", 1))

# Create initial condition
def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
# Create boundary condition
bc = fem.DirichletBC(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Initialize solution vector
uh = fem.Function(V)
uh.interpolate(initial_condition)
if MPI.COMM_WORLD.Get_rank() == 0:
    # Plot initial condition
    plt.figure()
    plt.title('Time: {}'.format(t))
    plt.plot(uh.x.array)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Define solution variable
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
D = 0.1  # Diffusion coefficient

# Define non-constant source term
def source_term(x, t):
    return np.sin(x[0]) * np.cos(x[1]) * np.exp(-t)

# Define forms for diffusion and source terms
a_diffusion = u * v * ufl.dx + dt * D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L_source = dt * fem.interpolate(fem.Constant(domain, 0), V) * v * ufl.dx

# Assemble the bilinear and linear forms
A_diffusion = fem.assemble_matrix(a_diffusion, bcs=[bc])
b_source = fem.assemble_vector(L_source)

# Create solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A_diffusion)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Time-stepping loop
for i in range(1, num_steps + 1):
    t += dt
    print("Time step:", i)

    # Update source term at each time step
    L_source = dt * fem.interpolate(fem.Constant(domain, source_term), V) * v * ufl.dx
    fem.assemble_vector(b_source, L_source)

    # Apply Dirichlet boundary condition to the vector
    fem.apply_lifting(b_source, [a_diffusion], [[bc]])
    fem.set_bc(b_source, [bc])

    # Solve linear problem
    solver.solve(b_source, uh.vector)

    if MPI.COMM_WORLD.Get_rank() == 0:
        # Plot the solution
        plt.figure()
        plt.title('Time: {}'.format(t))
        plt.plot(uh.x.array)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    # Update the solution at previous time step
    uh_prev = uh.copy()

# Close the XDMF file
