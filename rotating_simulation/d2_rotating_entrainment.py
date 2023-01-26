"""
Dedalus script simulating 3D rotationally-constrainted, compositionally-stabilized, entraining convection.
"""

import numpy as np
import dedalus.public as d2
import logging
from mpi4py import MPI
from dedalus.extras import flow_tools
import time
import pathlib
logger = logging.getLogger(__name__)

def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.
    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)

def global_noise(domain, seed=42, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    return noise_field

#Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)


# Parameters - load in from parameter file (Nx, Nz, Ra, Pr, etc)
from control_parameters import parameters
locals().update(parameters)

# Additional Parameters

kappaT = (Rayleigh * Prandtl)**(-1/2)
kappaC = tau * kappaT
kappaC_bg = tau_bg * kappaT
nu = (Rayleigh / Prandtl)**(-1/2)
omega = (Taylor * Prandtl / Rayleigh)**(1/2)
aspect = 1.5

cfl_safety = 0.2
max_timestep = 0.25
dt = max_timestep
dtype = np.float64
ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

# Bases

xbasis = d2.Fourier('x', Nx, interval=(0, aspect*Lz), dealias=dealias)
ybasis = d2.Fourier('y', Nx, interval=(0, aspect*Lz), dealias=dealias)
zbasis = d2.Chebyshev('z', Nz, interval=(0, Lz), dealias=dealias)
domain = d2.Domain([xbasis,ybasis,zbasis], grid_dtype=np.float64)


# Problem

problem = d2.IVP(domain, variables=['p', 'T', 'C', 'u','v','w', 'Tz', 'Cz','Ox', 'Oy'])
problem.meta['p', 'T', 'C', 'u','v','w']['z']['dirichlet'] = True
problem.parameters['Lz'] = Lz
problem.parameters['Lx'] = aspect*Lz
problem.parameters['Ly'] = aspect*Lz
problem.parameters['kappaT'] =  (Rayleigh * Prandtl)**(-1/2)
problem.parameters['kappaC'] = tau * kappaT
problem.parameters['kappaC_bg'] = tau_bg * kappaT
problem.parameters['nu'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['omega'] = (Taylor * Prandtl / Rayleigh)**(1/2)
problem.parameters['aspect'] = aspect
problem.parameters['inv_R'] = inv_R
problem.substitutions['plane_avg(A)'] = 'integ(A, "x","y")/(Lx*Ly)'
problem.substitutions['vol_avg(A)'] = 'integ(A, "x","y","z")/(Lx*Lz*Lz)'
problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + v * dy(A) + w * A_z)'
problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['vel_rms']  = 'sqrt(u**2 + v**2 +  w**2)'
problem.substitutions['KE'] = '(vel_rms**2)/2'
problem.substitutions['KE_flux'] = '0.5*w*(KE)'
problem.substitutions['Oz'] = '(dx(v) - dy(u))'
problem.substitutions['buoyancy'] = 'T - inv_R*C'

problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
problem.add_equation("Ox - dy(w) + dz(v) = 0")
problem.add_equation("Oy - dz(u) + dx(w) = 0")
problem.add_equation("dt(u) + nu*(dy(Oz) - dz(Oy)) + dx(p) - omega*v      = (v*Oz - w*Oy)")
problem.add_equation("dt(v) + nu*(dz(Ox) - dx(Oz)) + dy(p) + omega*u      = (w*Ox - u*Oz)")
problem.add_equation("dt(w) + nu*(dx(Oy) - dy(Ox)) + dz(p) - (T - inv_R*C)= (u*Oy - v*Ox)")
problem.add_equation("dt(C) - kappaC*Lap(C, Cz)      = -UdotGrad(C, Cz)",condition="(nx != 0) or  (ny != 0)")
problem.add_equation("dt(C) - kappaC_bg*Lap(C, Cz)   = -UdotGrad(C, Cz)",condition="(nx == 0) and (ny == 0)")
problem.add_equation("dt(T) - kappaT*Lap(T, Tz)      = -UdotGrad(T, Tz)")
problem.add_equation("Tz - dz(T) = 0")
problem.add_equation("Cz - dz(C) = 0")

# Zero flux bc
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
# Stress free bc
problem.add_bc("right(Ox) = 0")                    
problem.add_bc("left(Ox) = 0")
problem.add_bc("right(Oy) = 0")                    
problem.add_bc("left(Oy) = 0")

problem.add_bc("right(Tz) = -1")                    
problem.add_bc("left(T) = Lz")

problem.add_bc("right(Cz) = 0")                     
problem.add_bc("left(C) = Lz")               
problem.add_bc("integ_z(p) = 0", condition="(nx == 0) and (ny == 0)")



# Timestepping
# Build solver
solver = problem.build_solver(d2.timesteppers.SBDF2)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)
T = solver.state['T']
Tz = solver.state['Tz']
C = solver.state['C']
Cz = solver.state['Cz']

# Random perturbations, initialized globally for same results in parallel
noise = global_noise(domain, frac=0.5)

T['g'] = Lz - z 
C['g'] = Lz - z

noise.set_scales(dealias)
T.set_scales(dealias)
C.set_scales(dealias)

z_de = domain.grid(-1, scales=domain.dealias)

T['g'] += 1e-5*noise['g']
C['g'] *= one_to_zero(z_de, 0.5*Lz, width=0.05)
C['g'] += 0.5*zero_to_one(z_de, 0.5*Lz, width=0.05)

T.differentiate('z', out=Tz)
C.differentiate('z', out=Cz)

# Integration parameters
solver.stop_sim_time = 2e+4
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

#import matplotlib.pyplot as plt
#plt.plot(z_de.ravel(), C['g'][0,0,:], label='C')
#plt.plot(z_de.ravel(), T['g'][0,0,:], label='T')
#plt.legend()
#plt.savefig('IC_{}.png'.format(MPI.COMM_WORLD.rank))


# Analysis

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=100)
snapshots.add_task("interp(buoyancy, x={})".format(aspect*Lz/2),  name='buoyancy_yz')
snapshots.add_task("interp(buoyancy, y={})".format(aspect*Lz/2),  name="buoyancy_xz")
snapshots.add_task("interp(buoyancy, x={})".format(aspect*Lz),  name="buoyancy_yz_side")
snapshots.add_task("interp(buoyancy, y={})".format(aspect*Lz),  name="buoyancy_xz_side")
snapshots.add_task("interp(buoyancy, z={})".format(0.75*Lz),  name="buoyancy_xy")
snapshots.add_task("interp(buoyancy, z={})".format(0.5*Lz),  name="buoyancy_xy_mid")
snapshots.add_task("interp(buoyancy, z={})".format(0.975*Lz),  name="buoyancy_xy_near_top")

snapshots.add_task("interp(Oz, x={})".format(aspect*Lz/2),  name='vorticity_yz')
snapshots.add_task("interp(Oz, y={})".format(aspect*Lz/2),  name="vorticity_xz")
snapshots.add_task("interp(Oz, x={})".format(aspect*Lz),  name="vorticity_yz_side")
snapshots.add_task("interp(Oz, y={})".format(aspect*Lz),  name="vorticity_xz_side")
snapshots.add_task("interp(Oz, z={})".format(0.75*Lz),  name="vorticity_xy")
snapshots.add_task("interp(Oz, z={})".format(0.5*Lz),  name="vorticity_xy_mid")
snapshots.add_task("interp(Oz, z={})".format(0.975*Lz),  name="vorticity_xy_near_top")

snapshots.add_task("interp(w, z={})".format(0.75*Lz),  name="vertical_velocity")
snapshots.add_task("interp(0.5*(u*u + v*v + w*w), z={})".format(0.75*Lz),  name="kinetic_energy")


profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.25, max_writes=100)
profiles.add_task("plane_avg(T)", name="T")
profiles.add_task("plane_avg(w*T)", name="T_conv_flux")
profiles.add_task("plane_avg(-kappaT*Tz)", name="T_cond_flux")
profiles.add_task("plane_avg(Tz)", name="T_grad")
profiles.add_task("plane_avg(C)", name="C")
profiles.add_task("plane_avg(w*C)", name="C_conv_flux")
profiles.add_task("plane_avg(-kappaC_bg*Cz)", name="C_cond_flux")
profiles.add_task("plane_avg(Cz)", name="C_grad")
profiles.add_task("plane_avg(KE_flux)", name="F_KE")
profiles.add_task("plane_avg(w*(T - inv_R*C))", name="F_Buoyancy")
profiles.add_task("plane_avg(w * p)", name="F_p")
profiles.add_task("plane_avg(w**3 / 2)", name="F_KE_vert")
profiles.add_task("plane_avg(sqrt(u**2 + v**2 + w**2)/omega)", name='Ro_bulk')
profiles.add_task("plane_avg(sqrt(Ox**2 + Oy**2 + Oz**2)/omega)", name='Ro_vort')
profiles.add_task("plane_avg(sqrt((Oz)**2)/omega)", name='Ro_z_vort')
profiles.add_task("plane_avg(nu*(Ox**2 + Oy**2 + Oz**2))", name='Dissipation')
profiles.add_task("plane_avg(u**2 + v**2)", name='KE_parallel')
profiles.add_task("plane_avg(w**2)", name='KE_vert')
profiles.add_task("plane_avg(-nu*(u*Oy - v*Ox))", name='viscous_flux')


# Checkpoint
checkpoint = solver.evaluator.add_file_handler('checkpoint', wall_dt=3600, max_writes=1)
checkpoint.add_system(solver.state)


# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety, threshold=0.1, max_dt=max_timestep)
CFL.add_velocities(('u','v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(u*u + v*v + w*w)/nu", name='Re')
flow.add_property("sqrt(u*u + v*v + w*w)/omega", name='Ro_bulk')
flow.add_property("sqrt(Ox*Ox + Oy*Oy + Oz*Oz)/omega", name='Ro_bulk')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration - 1) % 10   == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' % (solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' % flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' % solver.iteration)
    logger.info('Sim end time: %f' % solver.sim_time)
    logger.info('Run time: %.2f sec' % (end_time - start_time))
    logger.info('Run time: %f cpu-hr' % ((end_time - start_time) / 60 / 60 * domain.dist.comm_cart.size))