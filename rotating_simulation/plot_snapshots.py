from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = '.'
data_dir    = 'snapshots'

# Read in additional plot arguments
start_fig   = 1
start_file  = 1
out_name    = 'snapshots_horiz'
n_files     = int(1e6)

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, out_name=out_name, start_file=start_file, n_files=n_files)
#plotter_kwargs = { 'col_inch' : float(args['--col_inch']), 'row_inch' : float(args['--row_inch']) }
plotter_kwargs = { 'col_inch' : 8, 'row_inch' : 8 }

plotter.setup_grid(num_rows=1, num_cols=2, **plotter_kwargs)
plotter.add_colormesh('vorticity xz', x_basis='x', y_basis='z', remove_x_mean=True)
plotter.add_colormesh('vorticity xy', x_basis='x', y_basis='y', remove_mean=True)
plotter.plot_colormeshes(start_fig=start_fig, dpi=200)
