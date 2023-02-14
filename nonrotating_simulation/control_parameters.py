from collections import OrderedDict

parameters = OrderedDict()
parameters['Nz'] = 512
parameters['Nx'] = 192
parameters['Lz'] = 2
parameters['aspect'] = 1.5

parameters['Rayleigh'] = 1e7
parameters['Prandtl'] = 0.5
parameters['Taylor'] = 1
parameters['tau'] = parameters['Prandtl']
parameters['tau_bg'] = 3e-3
parameters['inv_R'] = 10
parameters['dealias'] = 3/2
parameters['stop_sim_time'] = 10000

