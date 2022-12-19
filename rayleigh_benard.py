"""
Dedalus script simulating 3D rotationally-constrainted, compositionally-stabilized, entraining convection.
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
aspect, Lz = 4, 1
Nx, Nz = 256, 64
Rayleigh = 2e6
Prandtl = 0.5
Taylor = 1
tau = Prandtl
inv_R = 10
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, aspect), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nx, bounds=(0, aspect), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
bases = (xbasis, ybasis, zbasis)
horiz_bases = (xbasis, ybasis)
p = dist.Field(name='p', bases=bases)
T = dist.Field(name='T', bases=bases)
C = dist.Field(name='C', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=horiz_bases)
tau_T2 = dist.Field(name='tau_T2', bases=horiz_bases)
tau_C1 = dist.Field(name='tau_C1', bases=horiz_bases)
tau_C2 = dist.Field(name='tau_C2', bases=horiz_bases)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=horiz_bases)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=horiz_bases)


# Substitutions
kappaT = (Rayleigh * Prandtl)**(-1/2)
kappaC = tau * kappaT
nu = (Rayleigh / Prandtl)**(-1/2)
omega = (Taylor * Prandtl / Rayleigh)**(1/2)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + ez*lift(tau_T1) # First-order reduction
grad_C = d3.grad(C) + ez*lift(tau_C1) # First-order reduction

#for stress-free BCs
strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
shear_stress = ez@(strain_rate(z=Lz))

#Initial / Background profiles
T['g'] = T0 = 1 - z
C['g'] = C0 = 1 - z

T0_bot = T(z=0).evaluate()
C0_bot = C(z=0).evaluate()

#TODO: get this from Rayleigh number
F = -10

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, T, C, u, tau_p, tau_T1, tau_T2, tau_C1, tau_C2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - kappaT*div(grad_T) + lift(tau_T2) = - u@grad(T)")
problem.add_equation("dt(C) - kappaC*div(grad_C) + lift(tau_C2) = - u@grad(C)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - (T - inv_R*C)*ez - omega*(cross(ez,u)) + lift(tau_u2) = - u@grad(u)")
problem.add_equation("C(z=0) = C0_bot")
problem.add_equation("T(z=0) = T0_bot")
problem.add_equation("u(z=0) = 0")
problem.add_equation("ez@(grad(C)(z=Lz)) = 0")
problem.add_equation("ez@(grad(T)(z=Lz)) = F")
problem.add_equation("ez@(u(z=Lz)) = 0")
problem.add_equation("ex@shear_stress = 0")
problem.add_equation("ey@shear_stress = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
noise = dist.Field(name='noise', bases=bases)
noise.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
T['g'] += noise['g']

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(T - inv_R*C, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

