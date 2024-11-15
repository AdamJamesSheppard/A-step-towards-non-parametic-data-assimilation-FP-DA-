import os
import gc
import logging
import json
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
from dolfin import *
from scipy.interpolate import LinearNDInterpolator

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = "config.json"


def load_config(config_path):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, "r") as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


config = load_config(CONFIG_PATH)


def initialize_mpi_environment():
    """Initialize MPI and print rank-specific information."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        logger.info(f"Initialized MPI environment with {size} processes")
    return comm, rank


def setup_mesh(config):
    """Create a mesh for the domain."""
    try:
        mesh = BoxMesh(
            MPI.COMM_WORLD,
            Point(*config["domain"]["lower_bounds"]),
            Point(*config["domain"]["upper_bounds"]),
            *config["domain"]["resolution"]
        )
        logger.info("Mesh created successfully.")
        return mesh
    except Exception as e:
        logger.error(f"Error in mesh creation: {e}")
        raise


class LorenzSystem(UserExpression):
    """Represents the Lorenz system."""
    def eval(self, values, x):
        sigma, rho, beta = config["lorenz"]["sigma"], config["lorenz"]["rho"], config["lorenz"]["beta"]
        values[0] = sigma * (x[1] - x[0])
        values[1] = x[0] * (rho - x[2]) - x[1]
        values[2] = x[0] * x[1] - beta * x[2]

    def value_shape(self):
        return (3,)


class StateSpaceDistribution(UserExpression):
    """Interpolate an initial density."""
    def __init__(self, interpolant, **kwargs):
        self.interpolant = interpolant
        super().__init__(**kwargs)

    def eval(self, values, x):
        result = self.interpolant(*x)
        values[:] = result if not np.isnan(result) else 0


def generate_diffusion_vector():
    """Generate a diffusion vector based on user-defined configuration."""
    params = config["diffusion"]
    return np.random.normal(params["mean"], params["std"], 2).tolist() + [
        np.random.uniform(params["range"][0], params["range"][1])
    ]


def define_diffusion_matrices(dw):
    """Define diffusion and control matrices."""
    diffusion_matrix = Constant(
        ((config["intensity"] * dw[i] * dw[j] for j in range(3)) for i in range(3))
    )
    control_matrix = Constant(config["control_matrix"])
    return diffusion_matrix, control_matrix


def load_initial_data(file_path, mesh):
    """Load initial density data."""
    try:
        data = np.genfromtxt(file_path, delimiter=',')
        interpolant = LinearNDInterpolator(data[:, :2], data[:, 2])
        logger.info(f"Loaded initial density data from {file_path}")
        return interpolant
    except Exception as e:
        logger.error(f"Failed to load initial data: {e}")
        raise


def solve_fokker_planck(mesh, u_sol, u0, solver, plot_dir):
    """Solve the Fokker-Planck equation iteratively."""
    simulation_time = 0
    counter = 0
    progress = tqdm(total=int(config["simulation"]["total_time"] / config["simulation"]["time_step"]),
                    desc="Solving Fokker-Planck")

    while simulation_time <= config["simulation"]["total_time"]:
        counter += 1

        try:
            solver.solve()
            u0.assign(u_sol)
        except Exception as e:
            logger.error(f"Solver error at step {counter}: {e}")
            break

        if counter % config["output"]["save_interval"] == 0:
            if MPI.COMM_WORLD.Get_rank() == 0:
                os.makedirs(f"{plot_dir}/data", exist_ok=True)
                solution_data = u0.compute_vertex_values(mesh)
                np.save(f"{plot_dir}/data/solution_{counter}.npy", solution_data)

        simulation_time += config["simulation"]["time_step"]
        progress.update(1)

    progress.close()


def main():
    comm, rank = initialize_mpi_environment()

    # Setup simulation
    mesh = setup_mesh(config)
    dw = generate_diffusion_vector()
    diffusion_matrix, control_matrix = define_diffusion_matrices(dw)

    interpolant = load_initial_data(config["initial_distribution_path"], mesh)

    # Function spaces
    V = FunctionSpace(mesh, 'CG', 1)
    W = VectorFunctionSpace(mesh, 'CG', 1)
    lorenz_expr = LorenzSystem(degree=2)
    FX = interpolate(lorenz_expr, W)

    # Initial density
    u = TrialFunction(V)
    v = TestFunction(V)
    u0 = interpolate(
        StateSpaceDistribution(interpolant, element=V.ufl_element(), degree=2),
        V
    )

    # Fokker-Planck equation
    operator = dot(diffusion_matrix * grad(u), grad(v)) * dx - u * inner(FX, grad(v)) * dx
    fokker_planck_eq = (1.0 / config["simulation"]["time_step"]) * dot(u - u0, v) * dx + config["theta"] * operator

    u_sol = Function(V)
    problem = LinearVariationalProblem(lhs(fokker_planck_eq), rhs(fokker_planck_eq), u_sol)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "ilu"

    # Solve and plot
    solve_fokker_planck(mesh, u_sol, u0, solver, config["output"]["plot_dir"])

    # Clean up
    gc.collect()
    comm.Barrier()


if __name__ == "__main__":
    main()
