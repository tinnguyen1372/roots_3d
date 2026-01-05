from ssl import OP_CIPHER_SERVER_PREFERENCE
from tkinter import Y
from gprMax.gprMax import api
from gprMax.receivers import Rx
from tools.outputfiles_merge import merge_files
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 
from tools.plot_Ascan import mpl_plot as mpl_plot_Ascan
from gprMax.receivers import Rx
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import random
import os
import itertools
from PIL import Image


class Roots_Func():
    def __init__(self, args) -> None:
        self.gpu = getattr(args, 'gpu', [0])
        self.num_scan = getattr(args, 'num_scan', 72)
        self.resol = getattr(args, 'resol', 0.005)
        self.time_window = getattr(args, 'time_window', 40e-9)
        self.h5_file = getattr(args, 'h5file', 'test_1.h5')
        
        # Dimensions for the object to be read from H5
        self.x, self.y, self.z = 2, 1, 2
        self.confined_permittivity = getattr(args, 'confined_permittivity', 5.24)
        self.confined_conductivity = getattr(args, 'confined_conductivity', 0.001)
        
        self.roots_permittivity = getattr(args, 'roots_permittivity', [12, 12])
        self.roots_conductivity = getattr(args, 'roots_conductivity', [0.0002, 0.0002])
        self.fractal_box_seed = getattr(args, 'fractal_box_seed', 42)

    def run_circular_scan(self):
        self.input = 'circular_scan.in'
        pml_cells = 20
        pml_val = self.resol * pml_cells

        # Define the simulation domain
        sharp_domain = [3.0, 1.5, 3.0]
        domain_3d = [
            float(sharp_domain[0] + 2 * pml_val), 
            float(sharp_domain[1] + 2 * pml_val), 
            float(sharp_domain[2] + 2 * pml_val)
        ]

        # Generate material file for H5 objects
        self_mat_file = 'Object_materials.txt'
        with open(self_mat_file, 'w') as f:
            for i in range(len(self.roots_permittivity)):
                f.write(f'#material: {self.roots_permittivity[i]} {self.roots_conductivity[i]} 1 0 Object{i}\n')

        # The Python block inside the .in file handles the rotation logic
        config = f'''#title: Circular Root Scan
#domain: {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f}
#dx_dy_dz: {self.resol} {self.resol} {self.resol}
#time_window: {self.time_window}
#pml_cells: {pml_cells}

#material: 5.24 0.001 1 0 hete_soil
soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil
fractal_box: {pml_val:.3f} {pml_val:.3f} {pml_val:.3f} {domain_3d[0] - pml_val:.3f} 1.1 {domain_3d[2] - pml_val:.3f} 1.5 1 1 1 20 hete_soil my_fractal_box {self.fractal_box_seed}
#python:
from gprMax.input_cmd_funcs import *
import numpy as np

r_ant = 1.10
cx, cz = {domain_3d[0]}/2, {domain_3d[2]}/2
total_runs = {self.num_scan}

# Angle for this specific step
angle_tx = (2 * np.pi * (current_model_run - 1)) / total_runs
# Offset Rx by about 10cm along the arc to avoid overlap
angle_rx = angle_tx + (0.10 / r_ant) 

tx_x, tx_z = cx + r_ant * np.cos(angle_tx), cz + r_ant * np.sin(angle_tx)
rx_x, rx_z = cx + r_ant * np.cos(angle_rx), cz + r_ant * np.sin(angle_rx)
antenna_y = 1.20 + (2 * {self.resol})

waveform('gaussian', 1, 5e8, 'my_gaussian')
# Use 'z' as it is the best supported horizontal polarization
hertzian_dipole('z', tx_x, antenna_y, tx_z, 'my_gaussian')
rx(rx_x, antenna_y, rx_z)
#end_python:
#material: {self.confined_permittivity} {self.confined_conductivity} 1 0 confined_material
#box: {pml_val:.3f} 1.100 {pml_val:.3f} {domain_3d[0] - pml_val:.3f} 1.200 {domain_3d[2] - pml_val:.3f} confined_material
#geometry_objects_read: {(domain_3d[0]/2 - self.x/2):.3f} {domain_3d[1]/2 - self.y/2 - 0.25:.3f} {(domain_3d[2]/2 - self.z/2):.3f} {self.h5_file} {self_mat_file}
geometry_view: 0 0 0 {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f} {self.resol} {self.resol} {self.resol} CircularScan n
''' 
        with open(self.input, 'w') as f:
            f.write(config)

        # Run the simulation
        api(self.input, 
            n=int(self.num_scan), 
            gpu=[0],
            # geometry_only=True, 
            geometry_fixed=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_scan', type=int, default=36) # 5 degrees per step
    parser.add_argument('--h5file', type=str, default='test_1.h5')
    args = parser.parse_args()
    
    rootimg = Roots_Func(args=args)
    rootimg.run_circular_scan()