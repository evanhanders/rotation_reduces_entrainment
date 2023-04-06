"""
This script plots a 3D cutout view of a box simulation using PyVista and plotpal parallelization.

Usage:
    plot_3d_box_pyvista.py [options]

Options:
    --root_dir=<str>     Root directory path [default: ./]
    --data_dir=<dir>     Name of data handler directory [default: snapshots]
    --scale=<s>          resolution scale factor [default: 1]
    --start_file=<n>     start file number [default: 1]
"""
import gc
from collections import OrderedDict

from mpi4py import MPI
import h5py
import numpy as np
from docopt import docopt
args = docopt(__doc__)

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FormatStrFormatter
import pyvista as pv
import matplotlib.pyplot as plt

from plotpal.plot_grid import PyVista3DPlotGrid
from plotpal.volumes import construct_surface_dict
from plotpal.file_reader import SingleTypeReader, match_basis

# Read in master output directory
root_dir    = args['--root_dir']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

cmaps = ['PuOr_r']
fields = ['vorticity']
task_bases = ['yz side', 'xz side', 'xy near top', 'yz', 'xz', 'xy mid']
tasks = []
for f in fields:
    for tb in task_bases:
        tasks.append('{} {}'.format(f, tb))
yz_side = tasks[0]
xz_side = tasks[1]
xy_top  = tasks[2]

def get_minmax(field, cmap_exclusion=0.01, pos_def=False):

        vals = np.sort(field.flatten())
        if pos_def:
            vals = np.sort(vals)
            if np.mean(vals) < 0:
                vmin, vmax = vals[int(cmap_exclusion*len(vals))], 0
            else:
                vmin, vmax = 0, vals[int((1-cmap_exclusion)*len(vals))]
        else:
            vals = np.sort(np.abs(vals))
            vmax = vals[int((1-cmap_exclusion)*len(vals))]
            vmin = -vmax

        if vmin is not None:
            vmin = vmin
        if vmax is not None:
            vmax = vmax

        return vmin, vmax

fig_name='volume_visualization'
plotter = SingleTypeReader(root_dir, data_dir, fig_name, start_file=int(args['--start_file']), n_files=np.inf, distribution='even-write')
if not plotter.idle:
    first = True

    #Pyvista setup
    view = 0
    size = int(1000*float(args['--scale']))
    grid = PyVista3DPlotGrid(num_rows=1, num_cols=1, size=size)
    pv_grids = []

    sargs = dict(
        title_font_size=int(size/50),
        label_font_size=int(size/60),
        shadow=True,
        n_labels=5,
        italic=True,
        fmt="%.2f",
        font_family="arial",
        color='black'
    )


    data_dicts = []
    while plotter.writes_remain():
        dsets, ni = plotter.get_dsets(tasks)
        time_data = plotter.current_file_handle['scales']

        for ind, field in enumerate(fields):
            grid.change_focus_single(ind) #index grid number -- useful if you wanna make multiple plots.
            grid.pl.set_background('white', all_renderers=False)
        
            #Only get grid info on first pass
            if first:
                x = match_basis(dsets[xy_top], 'x')
                y = match_basis(dsets[xy_top], 'y')
                z = match_basis(dsets[xz_side], 'z')
                Lx = x.max() - x.min()
                Ly = y.max() - y.min()
                Lz = z.max() - z.min()
                x_mid = x.min() + 0.5*Lx
                y_mid = y.min() + 0.5*Ly
                z_mid = z.min() + 0.5*Lz

            left_field, right_field, top_field, mid_left_field, mid_right_field, mid_top_field = [np.squeeze(dsets[f][ni,:]) for f in tasks[ind*6:(ind+1)*6]]
            xy_side = construct_surface_dict(x, y, z.max(), top_field,    x_bounds=(x.min(), 1.01*x_mid), y_bounds=(y.min(), 1.01*y_mid))
            xz_side = construct_surface_dict(x, y.max(), z, right_field,  x_bounds=(x.min(), 1.01*x_mid), z_bounds=(z.min(), 1.01*z_mid))
            yz_side = construct_surface_dict(x.max(), y, z, left_field,   y_bounds=(y.min(), 1.01*y_mid), z_bounds=(z.min(), 1.01*z_mid))
            xy_mid = construct_surface_dict(x, y, z_mid, mid_top_field,   x_bounds=(0.95*x_mid, x.max()), y_bounds=(0.95*y_mid, y.max()), bool_function=np.logical_and)
            xz_mid = construct_surface_dict(x, y_mid, z, mid_right_field, x_bounds=(0.95*x_mid, x.max()), z_bounds=(0.95*z_mid, z.max()), bool_function=np.logical_and)
            yz_mid = construct_surface_dict(x_mid, y, z, mid_left_field,  y_bounds=(0.95*y_mid, y.max()), z_bounds=(0.95*z_mid, z.max()), bool_function=np.logical_and)
            side_list = [xy_side, xz_side, yz_side, xy_mid, xz_mid, yz_mid]

            cmap = matplotlib.cm.get_cmap(cmaps[ind])
            vmin, vmax = get_minmax(left_field, cmap_exclusion=0.01)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            for i, d in enumerate(side_list):
                if first:
                    grid.pl.set_background('white', all_renderers=False)
                    pv_grids.append(pv.StructuredGrid(d['x'], d['y'], d['z']))
                    pv_grids[i][field] = np.array(d['surfacecolor'].flatten(order='F'))
                    grid.pl.add_mesh(pv_grids[i], scalars=field, cmap = cmap, clim = [vmin, vmax], scalar_bar_args={'color' : 'black'})
                else:
                    pv_grids[i][field] = np.array(d['surfacecolor'].flatten(order='F'))

            camera_distance = 3
            grid.pl.camera.position = np.array((Lx, Ly, Lz))*camera_distance


            if not first:
                grid.pl.update(force_redraw=True)
                grid.pl.update_scalar_bar_range([vmin, vmax])

       # Save figure
        savepath = '{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, time_data['write_number'][ni])
        grid.save(savepath)
        first = False
        gc.collect()




