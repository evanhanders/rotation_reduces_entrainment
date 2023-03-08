import dedalus
from collections import OrderedDict
import h5py
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import plot_mpl, init_notebook_mode
from plotpal.file_reader import SingleTypeReader, match_basis
from scipy.interpolate import interp1d
import matplotlib

def construct_surface_dict(x_vals, y_vals, z_vals, data_vals, x_bounds=None, y_bounds=None, z_bounds=None, bool_function=np.logical_or):
    """
    Takes grid coordinates and data on grid and prepares it for 3D surface plotting in plotly
    
    Arguments:
    x_vals : NumPy array (1D) or float
        Gridspace x values of the data
    y_vals : NumPy array (1D) or float
        Gridspace y values of the data
    z_vals : NumPy array (1D) or float
        Gridspace z values of the data
    data_vals : NumPy array (2D)
        Gridspace values of the data
        
    Keyword Arguments:
    x_bounds : Tuple of floats of length 2
        If specified, the min and max x values to plot
    y_bounds : Tuple of floats of length 2
        If specified, the min and max y values to plot
    z_bounds : Tuple of floats of length 2
        If specified, the min and max z values to plot
        
    Returns a dictionary of keyword arguments for plotly's surface plot function
    
    """
    if type(x_vals) == np.ndarray and type(y_vals) == np.ndarray :
        yy, xx = np.meshgrid(y_vals, x_vals)
        zz = z_vals * np.ones_like(xx)
    elif type(x_vals) == np.ndarray and type(z_vals) == np.ndarray :
        zz, xx = np.meshgrid(z_vals, x_vals)
        yy = y_vals * np.ones_like(xx)    
    elif type(y_vals) == np.ndarray and type(z_vals) == np.ndarray :
        zz, yy = np.meshgrid(z_vals, y_vals)
        xx = x_vals * np.ones_like(yy)  
    
    if x_bounds is None:
        if type(y_vals) == np.ndarray and type(z_vals) == np.ndarray and bool_function == np.logical_or :
            x_bool = np.zeros_like(yy)
        else:
            x_bool = np.ones_like(yy)
    else:
        x_bool = (xx >= x_bounds[0])*(xx <= x_bounds[1])
        
    if y_bounds is None:
        if type(x_vals) == np.ndarray and type(z_vals) == np.ndarray and bool_function == np.logical_or :
            y_bool = np.zeros_like(xx)
        else:
            y_bool = np.ones_like(xx)
    else:
        y_bool = (yy >= y_bounds[0])*(yy <= y_bounds[1])

    if z_bounds is None:
        if type(x_vals) == np.ndarray and type(y_vals) == np.ndarray and bool_function == np.logical_or :
            z_bool = np.zeros_like(xx)
        else:
            z_bool = np.ones_like(xx)
    else:
        z_bool = (zz >= z_bounds[0])*(zz <= z_bounds[1])
        
    
    side_bool = bool_function.reduce((x_bool, y_bool, z_bool)) 

        
    side_info = OrderedDict()
    side_info['x'] = np.where(side_bool, xx, np.nan)
    side_info['y'] = np.where(side_bool, yy, np.nan)
    side_info['z'] = np.where(side_bool, zz, np.nan)
    side_info['surfacecolor'] = np.where(side_bool, data_vals, np.nan)
    
    return side_info

from scipy.interpolate import RegularGridInterpolator as RGI

fields = ['vorticity']

px_base = 500
fontsize = int(16 * px_base/500)
x_pix, y_pix = px_base, px_base
cmaps = ['PuOr_r']
#colorbar_x = [0.45, 0.97]
colorbar_dict=dict(lenmode='fraction', thicknessmode = 'fraction', len=0.25, thickness=0.02)
fig = go.Figure(layout={'width': x_pix, 'height': y_pix})
fig = make_subplots(rows=1, cols=1, specs=[[{'is_3d': True},]], subplot_titles=field_names, horizontal_spacing=0.025)
scene_dict = {          'xaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
                        'yaxis': {'showbackground':False, 'tickvals':[], 'title':''}, 
                        'zaxis': {'showbackground':False, 'tickvals':[], 'title':''} }
fig.update_layout(scene = scene_dict,
                      margin={'l':0, 'r': 0, 'b':0, 't':0, 'pad':0}, 
                      font={'size' : fontsize, 'family' : 'Times New Roman'}, 
                      annotations={'font' : {'size' : fontsize, 'family' : 'Times New Roman'}})

max_boost = 0.4
cbar_ticks = []
#I change this for every file (inocent dog face)
for file in ["./snapshots/non_rotating/snapshots_s8.h5"]:
    def make_3d_box(plot_ind,file):
        for ind, field in enumerate(fields):
            with h5py.File(file, 'r') as f:
            
                mean = 0*np.mean(f['tasks']['{} yz side'.format(field)][plot_ind,:].squeeze(), axis=0)
                yz_side_data=f['tasks']['{} yz side'.format(field)][plot_ind,:].squeeze() - mean

                mean = 0*np.mean(f['tasks']['{} yz'.format(field)][plot_ind,:].squeeze(), axis=0)
                yz_mid_data= f['tasks']['{} yz'.format(field)][plot_ind,:].squeeze() - mean

                mean = 0*np.mean(f['tasks']['{} xz side'.format(field)][plot_ind,:].squeeze(), axis=0)
                xz_side_data= f['tasks']['{} xz side'.format(field)][plot_ind,:].squeeze() - mean

                mean = 0*np.mean(f['tasks']['{} xz'.format(field)][plot_ind,:].squeeze(), axis=0)
                xz_mid_data= f['tasks']['{} xz'.format(field)][plot_ind,:].squeeze() - mean

                mean = 0*np.mean(f['tasks']['{} xy near top'.format(field)][plot_ind,:].squeeze())
                xy_side_data= f['tasks']['{} xy near top'.format(field)][plot_ind,:].squeeze()-mean

                mean = 0*np.mean(f['tasks']['{} xy mid'.format(field)][plot_ind,:].squeeze())
                xy_mid_data= f['tasks']['{} xy mid'.format(field)][plot_ind,:].squeeze() - mean
                
                
                #This could change for your run
                
                x = f['scales']['x_hash_ea83cf3c8dc6521f5bb4237afe0ace05c278fff6'][()].squeeze()
                y = f['scales']['y_hash_ea83cf3c8dc6521f5bb4237afe0ace05c278fff6'][()].squeeze()
                z = f['scales']['z_hash_423c9d8e34a53d507027e5295ba685a53e4fafe2'][()].squeeze()
            
        
            x_max, x_min = (x.max(), x.min())
            y_max, y_min = (y.max(), y.min())
            z_max, z_mid, z_min = (1.95, 1, z.min())

            #indexing works for fourier; midpoint calc good for chebyshev
            x_mid = x[int(len(x)/2)]#x_min + (x_max - x_min)/2
            y_mid = y[int(len(x)/2)]#y_min + (y_max - y_min)/2

            x_mid_off = x_mid + 1e-3
            y_mid_off = y_mid + 1e-3
            z_mid_off = z_mid + 1e-3

            #Construct 1D outline lines
            lines = []
            constX_xvals = (x_min, x_max, x_mid, x_max, x_max, x_mid_off)
            constX_zvals = (z_max, z_max, z_max, z_min, z_mid, z_mid_off)
            constX_ybounds = ([y_min, y_max], [y_min, y_mid], [y_mid, y_max], [0, y_max], [y_mid, y_max], [y_mid, y_max])
            for x_val, y_bounds, z_val in zip(constX_xvals, constX_ybounds, constX_zvals):
                lines.append(OrderedDict())
                lines[-1]['y'] = np.linspace(*tuple(y_bounds), 2)
                lines[-1]['x'] = x_val*np.ones_like(lines[-1]['y'])
                lines[-1]['z'] = z_val*np.ones_like(lines[-1]['y'])

            constY_yvals = (y_min, y_max, y_mid, y_max, y_max, y_mid_off)
            constY_zvals = (z_max, z_max, z_max, z_min, z_mid, z_mid_off)
            constY_xbounds = ([x_min, x_max], [x_min, x_mid], [x_mid, x_max], [x_min, x_max], [x_mid, x_max], [x_mid, x_max])
            for x_bounds, y_val, z_val in zip(constY_xbounds, constY_yvals, constY_zvals):
                lines.append(OrderedDict())
                lines[-1]['x'] = np.linspace(*tuple(x_bounds), 2)
                lines[-1]['y'] = y_val*np.ones_like(lines[-1]['x'])
                lines[-1]['z'] = z_val*np.ones_like(lines[-1]['x'])

            constZ_xvals = (x_min, x_max, x_mid_off, x_mid, x_max)
            constZ_yvals = (y_max, y_min, y_mid_off, y_max, y_mid)
            constZ_zbounds = ([z_min, z_max], [z_min, z_max], [z_mid, z_max], [z_mid, z_max], [z_mid, z_max])
            for x_val, y_val, z_bounds in zip(constZ_xvals, constZ_yvals, constZ_zbounds):
                lines.append(OrderedDict())
                lines[-1]['z'] = np.linspace(*tuple(z_bounds), 2)
                lines[-1]['x'] = x_val*np.ones_like(lines[-1]['z'])
                lines[-1]['y'] = y_val*np.ones_like(lines[-1]['z'])


            xy_buff = 1e-3
            z_buff = 1e-3
            #make sure that x_mid, y_mid, z_mid are present in the data
            if x_mid+xy_buff not in x:
                x_new = np.sort(np.append(x, x_mid+xy_buff))
            else:
                x_new = x
            if y_mid+xy_buff not in y:
                y_new = np.sort(np.append(y, y_mid+xy_buff))
            else:
                y_new = y
            if z_mid+z_buff not in z:
                z_new = np.sort(np.append(z, z_mid+z_buff))
            else:
                z_new = z

            xy_yy, xy_xx = np.meshgrid(y_new, x_new)
            xz_zz, xz_xx = np.meshgrid(z_new, x_new)
            yz_zz, yz_yy = np.meshgrid(z_new, y_new)

            xy_side_data = RGI(points=(x,y), values=xy_side_data, method='linear')((xy_xx, xy_yy))
            xz_side_data = RGI((x,z), xz_side_data, method='linear')((xz_xx, xz_zz))
            yz_side_data = RGI((y,z), yz_side_data, method='linear')((yz_yy, yz_zz))

            xy_mid_data = RGI((x,y), xy_mid_data, method='linear')((xy_xx, xy_yy))
            xz_mid_data = RGI((x,z), xz_mid_data, method='linear')((xz_xx, xz_zz))
            yz_mid_data = RGI((y,z), yz_mid_data, method='linear')((yz_yy, yz_zz))

            x = x_new
            y = y_new
            z = z_new

            xy_side = construct_surface_dict(x, y, z_max, xy_side_data, x_bounds=(x_min, x_mid+xy_buff), y_bounds=(y_min, y_mid+xy_buff))
            xz_side = construct_surface_dict(x, y_max, z, xz_side_data, x_bounds=(x_min, x_mid+xy_buff), z_bounds=(z_min, z_mid+z_buff))
            yz_side = construct_surface_dict(x_max, y, z, yz_side_data, y_bounds=(y_min, y_mid+xy_buff), z_bounds=(z_min, z_mid+z_buff))

            yz_mid = construct_surface_dict(x_mid, y, z, yz_mid_data, y_bounds=(y_mid, y_max), z_bounds=(z_mid, z_max + z_buff), bool_function=np.logical_and)
            xy_mid = construct_surface_dict(x, y, z_mid, xy_mid_data, x_bounds=(x_mid, x_max), y_bounds=(y_mid, y_max), bool_function=np.logical_and)
            xz_mid = construct_surface_dict(x, y_mid, z, xz_mid_data, x_bounds=(x_mid, x_max), z_bounds=(z_mid + z_buff, z_max), bool_function=np.logical_and)

            for d in [xz_side, xz_mid, yz_side, yz_mid]:
                for key in ['x', 'y', 'surfacecolor', 'z']:
                    d[key][d['z'] > z_max] = np.nan
            surface_dicts = [xz_side, yz_side, yz_mid, xz_mid, xy_mid, xy_side]

            tickvals = ticktext = None

            colorbar_dict['tickvals'] = tickvals
            colorbar_dict['ticktext'] = ticktext
            colorbar_dict['outlinecolor'] = 'black'
            colorbar_dict['xanchor'] = 'center'
            colorbar_dict['tickfont'] = {'family' : "Times New Roman"}
            colorbar_dict['outlinecolor'] = 'black'
            colorbar_dict['outlinewidth'] = 3
            for d in surface_dicts:
                d['cmin'] = -0.5*np.nanmax(xz_side['surfacecolor'])
                d['cmax'] = 0.5*np.nanmax(xz_side['surfacecolor'])
                d['colorbar'] = colorbar_dict
                d['colorscale'] = cmaps[ind]
                d['showscale'] = False
                d['lighting'] = {'ambient' : 1}


            for surface_dict in surface_dicts:
                fig.add_trace(go.Surface(**surface_dict),  1, 1)

            for line_dict in lines:
                fig.add_trace(go.Scatter3d(**line_dict, mode='lines',line={'color':'black'},line_width=4, showlegend=False), 1, 1)

            for anno in fig['layout']['annotations']:
                anno['text'] = ''



            #normalized from 0 or -1 to 1; not actually in data x, y, z units.
            viewpoint = {'camera_eye' : {'x' : 1.3*1.1, 'y': 1.3*1.1, 'z' : 0.9*1.1}
                        }
            fig.update_layout(scene = viewpoint)
            # Since I run this for a single file, I have to add a number to the frame name so it does not overwrite
            pio.write_image(fig, './frames/writes_%06i.png'%(plot_ind+1 + 700), width=x_pix, height=y_pix, format='png', engine='kaleido')
            fig.data = [] # this is important to clear the figure, otherwise it is very slow.
        
    # I loop here for my 100 writes per file
    for plot_ind in range(0,100):
        print(plot_ind)
        execute = make_3d_box(plot_ind,file)     