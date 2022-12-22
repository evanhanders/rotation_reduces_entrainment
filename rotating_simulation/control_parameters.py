from collections import OrderedDict

parameters = OrderedDict()
parameters['Nz'] = 256
parameters['Nx'] = 128
parameters['Lz'] = 2
parameters['aspect'] = 0.5

parameters['Rayleigh'] = 1e8
parameters['Prandtl'] = 0.5
parameters['Taylor'] = 1e10
parameters['tau'] = parameters['Prandtl']
parameters['tau_bg'] = 4e-3
parameters['inv_R'] = 3
parameters['dealias'] = 3/2
parameters['stop_sim_time'] = 1e4

