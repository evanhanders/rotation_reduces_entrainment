"""
Dedalus script simulating 3D rotationally-constrainted, compositionally-stabilized, entraining convection.
"""

import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)

#Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)


# Parameters
aspect, Lz = 1, 2
Nx, Nz = 4, 64
Rayleigh = 1e8
Prandtl = 0.5
Taylor = 1
tau = Prandtl
tau_bg = 1e-3
inv_R = 3
dealias = 3/2
stop_sim_time = 100
timestepper = d3.SBDF2
max_timestep = 0.125
dtype = np.float64

ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, aspect*Lz), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nx, bounds=(0, aspect*Lz), dealias=dealias)
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
kappaC_bg = tau_bg * kappaT
nu = (Rayleigh / Prandtl)**(-1/2)
omega = (Taylor * Prandtl / Rayleigh)**(1/2)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
x_de, y_de, z_de = dist.local_grids(xbasis, ybasis, zbasis, scales=(dealias, dealias, dealias))
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + ez*lift(tau_T1) # First-order reduction
grad_C = d3.grad(C) + ez*lift(tau_C1) # First-order reduction

# operators

grad = lambda A: d3.Gradient(A, coords)
dot = d3.DotProduct
vorticity = d3.curl(u)

FK_vert = dot(ez,u)**2
FK_parallel = dot(ex,u)**2 + dot(ey,u)**2

#for stress-free BCs
strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
shear_stress = ez@(strain_rate(z=Lz))

#Initial / Background profiles
T['g'] = T0 = (Lz - z)
C['g'] = C0 = (Lz - z)

T0_bot = T(z=0).evaluate()
C0_bot = C(z=0).evaluate()
T0z_top = (ez@(d3.grad(T)(z=Lz))).evaluate()

#TODO: get this from Rayleigh number

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, T, C, u, tau_p, tau_T1, tau_T2, tau_C1, tau_C2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - kappaT*div(grad_T) + lift(tau_T2) = - (u@grad(T))")
problem.add_equation("dt(C) - kappaC*div(grad_C)    + lift(tau_C2) = - (u@grad(C))", condition="(nx != 0) or  (ny != 0)")
problem.add_equation("dt(C) - kappaC_bg*div(grad_C) + lift(tau_C2) = - (u@grad(C))", condition="(nx == 0) and (ny == 0)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - (T - inv_R*C)*ez - omega*(cross(ez,u)) + lift(tau_u2) = - cross(vorticity,u)")
problem.add_equation("C(z=0) = C0_bot")
problem.add_equation("T(z=0) = T0_bot")
problem.add_equation("u(z=0) = 0")
problem.add_equation("ez@(grad(C)(z=Lz)) = 0")
problem.add_equation("ez@(grad(T)(z=Lz)) = T0z_top")
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

noise.change_scales(dealias)
T.change_scales(dealias)
C.change_scales(dealias)
T['g'] += noise['g']

C['g'] *= one_to_zero(z_de, 0.5*Lz, width=0.05*Lz)
C['g'] += 0.5*zero_to_one(z_de, 0.5*Lz, width=0.05*Lz)

import matplotlib.pyplot as plt
plt.plot(z_de.ravel(), C['g'][0,0,:], label='C')
plt.plot(z_de.ravel(), T['g'][0,0,:], label='T')
plt.legend()
plt.show()

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task((T - inv_R*C)(x=aspect*Lz/2), name='buoyancy yz')
snapshots.add_task((T - inv_R*C)(y=aspect*Lz/2), name='buoyancy xz')
snapshots.add_task((T - inv_R*C)(z=0.75*Lz), name='buoyancy xy')

snapshots.add_task((ez@d3.curl(u))(x=aspect*Lz/2), name='vorticity yz')
snapshots.add_task((ez@d3.curl(u))(y=aspect*Lz/2), name='vorticity xz')
snapshots.add_task((ez@d3.curl(u))(z=0.75*Lz), name='vorticity xy')

snapshots.add_task((ez@u)(z=0.75*Lz), name='vertical velocity')
snapshots.add_task((0.5*u@u)(z=0.75*Lz), name='kinetic energy')



plane_avg = lambda A: d3.Integrate(d3.Integrate(A, coords['x']),coords['y'])
profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.1, max_writes=50)
profiles.add_task(plane_avg(T), name='T')
profiles.add_task(plane_avg(dot(ez, u*T)), name='T_conv_flux')
profiles.add_task(plane_avg(dot(ez, -kappaT*grad(T))), name='T_cond_flux')
profiles.add_task(plane_avg(dot(ez, grad(T))), name='T_grad')
profiles.add_task(plane_avg(C), name='C')
profiles.add_task(plane_avg(dot(ez, u*C)), name='C_conv_flux')
profiles.add_task(plane_avg(dot(ez, -kappaC_bg*grad(C))), name='C_cond_flux')
profiles.add_task(plane_avg(dot(ez, grad(C))), name='C_grad')
profiles.add_task(plane_avg(dot(ez, 0.5*u*dot(u,u))), name='KE_flux')
profiles.add_task(plane_avg(dot(u,(T - inv_R*C)*ez)), name='Buoyancy_flux')
profiles.add_task(plane_avg(nu*dot(vorticity,vorticity)), name='Dissipation')
profiles.add_task(plane_avg(dot(ez,u*p)), name='pressure_flux')
profiles.add_task(plane_avg(-nu*dot(ez,d3.cross(u,vorticity))), name='viscous_flux')
profiles.add_task(plane_avg(FK_vert), name='KE_vert')
profiles.add_task(plane_avg(FK_parallel), name='KE_parallel')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.25, threshold=0.05,
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

