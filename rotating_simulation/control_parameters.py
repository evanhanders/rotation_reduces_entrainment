from collections import OrderedDict

parameters = OrderedDict()
parameters['Nz'] = 512
parameters['Nx'] = 128
parameters['Lz'] = 2
parameters['aspect'] = 1 

parameters['Rayleigh'] = 1e7
parameters['Prandtl'] = 0.5
parameters['Taylor'] = 3e7
parameters['tau'] = parameters['Prandtl']
parameters['tau_bg'] = 1e-3
parameters['inv_R'] = 10 
parameters['dealias'] = 3/2
parameters['stop_sim_time'] = 5e4

