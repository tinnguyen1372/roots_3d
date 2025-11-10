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
        self.gpu = getattr(args, 'gpu', 0)

        self.num_scan = getattr(args, 'num_scan', 72)
        self.resol = getattr(args, 'resol', 0.005)
        self.time_window = getattr(args, 'time_window', 30e-9)

        # Geometry parameters
        self.h5_file = getattr(args, 'h5file', 'test.h5')
        self.x , self.y , self.z = 2 , 1 , 2
        # self.pix = int(max(self.x, self.y, self.z)/self.resol)

        self.confined_permittivity = getattr(args, 'confined_permittivity', 5.24)
        self.confined_conductivity = getattr(args, 'confined_conductivity', 0.001)
        
        self.roots_permittivity = getattr(args, 'roots_permittivity', [24, 24])
        self.roots_conductivity = getattr(args, 'roots_conductivity', [0.0002, 0.0002])

        self.src_to_gnd = 0.1 
        self.src_to_rx = 0.1 
        self.confined_size = 20 * self.resol

        self.fractal_box_seed = getattr(args, 'fractal_box_seed', 42) # random_randint(0,100)

    def run_circular_scan(self):
        self.input = 'circular_scan.in'
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.05

        sharp_domain = 3 , 1.5, 3
        domain_3d = [
            float(sharp_domain[0] + 2 * pml), 
            float(sharp_domain[1] + 2 * pml), 
            float(sharp_domain[2] + 2 * pml)
        ]
        self_mat_file = 'Object_materials.txt'
        with open(self_mat_file, 'w') as f:
            for i in range(len(self.roots_permittivity)):
                f.write('#material: {} {} 1 0 Object{}\n'.format(self.roots_permittivity[i], self.roots_conductivity[i], i))
            f.close()

        data = []
        self.input = f'circular_scan.in'
        config = f'''
#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f}
#dx_dy_dz: {self.resol} {self.resol} {self.resol}
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells}

Environment
#soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil
#fractal_box: {pml:.3f} {pml:.3f} {pml:.3f} {domain_3d[0] - pml:.3f} {1:.3f} {domain_3d[2] - pml:.3f} 1.5 1 1 1 20 hete_soil my_fractal_box {self.fractal_box_seed}
#material: {self.confined_permittivity} {self.confined_conductivity} 1 0 confined_material
#box: {pml:.3f} {1:.3f} {pml:.3f} {domain_3d[0] - pml:.3f} {1.15:.3f} {domain_3d[2] - pml:.3f} confined_material

#python:
from gprMax.input_cmd_funcs import *
import numpy as np
r = 1.5         # radius
delta = 0.005   # grid resolution
theta = np.linspace(0, 2*np.pi, number_model_runs+1)

# continuous circle
cx = 3.200 / 2
cy = 3.200 / 2
x = cx + r * np.cos(theta)
y = cy + r * np.sin(theta)
# quantized coordinates
xq = np.round(x / delta) * delta
yq = np.round(y / delta) * delta

# remove duplicates (optional)
points = np.unique(np.column_stack((xq, yq)), axis=0)

waveform('gaussian', 1, 5e8, 'my_gaussian')
hertzian_dipole('y', points[current_model_run-1][0], 1.25, points[current_model_run-1][1], 'my_gaussian') 
rx(
    points[current_model_run-2][0],
    1.25,
    points[current_model_run-2][1]
)
#end_python:
#geometry_objects_read: {(domain_3d[0]/2 - self.x/2) :.3f} {domain_3d[1]/2 - self.y/2 - 0.25:.3f} {(domain_3d[2]/2 - self.z/2):.3f} {self.h5_file} Object_materials.txt

    '''     
        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        api(self.input, 
            n=int(self.num_scan), 
            gpu=[0],
            geometry_only=False, geometry_fixed=False)
            # merge_files(self.input)
            # data_quarter = get_output_data(self.input)
            # bscan = mpl_plot_Bscan(data, self.resol)
            # ascan = mpl_plot_Ascan(dat
            # a, self.resol)
            # return bscan, ascan
            # data.append(data_quarter)
        # INSERT_YOUR_CODE
        # Merge the 4 data_quarter (assumed to be numpy arrays or similar)
        # import numpy as np
        # merged_data = np.concatenate(data, axis=0)

        # plt.imshow(merged_data, cmap='gray', aspect='auto')
        # plt.savefig('merged_data.png')
        # plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roots Scanning for Through Imaging")      
    parser.add_argument('--start', type=int, default=0, help='Start of the generated geometry')
    parser.add_argument('--end', type=int, default=1, help='End of the generated geometry')
    parser.add_argument('--num_scan', type=int, default=72, help='Number of A-Scans')

    args = parser.parse_args()
    rootimg = Roots_Func(args=args)
    rootimg.run_circular_scan()