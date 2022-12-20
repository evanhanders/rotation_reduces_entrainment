"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_rolled_profiles.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: profiles]
    --subdir_name=<subdir_name>               Name of figure output directory & base name of saved figures [default: rolled_profiles]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --roll_writes=<int>                 Number of writes over which to take average
    --dpi=<dpi>                         Image pixel density [default: 200]
    --col_inch=<in>                    Figure width (inches) [default: 6]
    --row_inch=<in>                   Figure height (inches) [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.profiles import RolledProfilePlotter
from plotpal.file_reader import match_basis

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_file  = int(args['--start_file'])
subdir_name    = args['--subdir_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

roll_writes = args['--roll_writes']
if roll_writes is not None:
    roll_writes = int(roll_writes)

def T_fluxes(ax, dictionary, index):
    z = match_basis(dictionary['T_conv_flux'], 'z')
    conv_flux = dictionary['T_conv_flux'][index].ravel()
    cond_flux = dictionary['T_cond_flux'][index].ravel()
    tot_flux = conv_flux + cond_flux

    ax.plot(z, conv_flux, color='orange', label='convective')
    ax.plot(z, cond_flux, color='green', label='conductive')
    ax.plot(z, tot_flux, color='black')
    ax.legend(loc='upper left')
    ax.set_ylabel('T flux')
    ax.set_xlabel('z')

def C_fluxes(ax, dictionary, index):
    z = match_basis(dictionary['C_conv_flux'], 'z')
    conv_flux = dictionary['C_conv_flux'][index].ravel()
    cond_flux = dictionary['C_cond_flux'][index].ravel()
    tot_flux = conv_flux + cond_flux

    ax.plot(z, conv_flux, color='orange', label='convective')
    ax.plot(z, cond_flux, color='green', label='conductive')
    ax.plot(z, tot_flux, color='black')
    ax.legend(loc='upper left')
    ax.set_ylabel('C flux')
    ax.set_xlabel('z')




# Create Plotter object, tell it which fields to plot
plotter = RolledProfilePlotter(root_dir, file_dir=data_dir, out_name=subdir_name, roll_writes=roll_writes, start_file=start_file, n_files=n_files)
plotter.setup_grid(num_rows=2, num_cols=1, col_inch=float(args['--col_inch']), row_inch=float(args['--row_inch']))
#plotter.add_line('z', 'b', grid_num=0)
plotter.add_line('z', T_fluxes, grid_num=0, needed_tasks=['T_conv_flux','T_cond_flux'])
plotter.add_line('z', C_fluxes, grid_num=1, needed_tasks=['C_conv_flux','C_cond_flux'])
plotter.plot_lines()
