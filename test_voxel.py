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

        self.num_scan = getattr(args, 'num_scan', 80)
        self.resol = getattr(args, 'resol', 0.005)
        self.time_window = getattr(args, 'time_window', 30e-9)

        # Geometry parameters
        self.h5_file = getattr(args, 'h5file', 'test_voxel.h5')
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

    def run_straight_scan(self):
        self.input = 'straight_scan.in'
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
        for quarter in range(1):
            self.input = f'straight_scan_{quarter+1}.in'
            config = f'''
#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f}
#dx_dy_dz: {self.resol} {self.resol} {self.resol}
#time_window: {self.time_window}
#waveform: ricker 1 500e6 my_wave
#pml_cells: {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells}

Environment
#material: {self.confined_permittivity} {self.confined_conductivity} 1 0 confined_material
#box: {pml:.3f} {1.1:.3f} {pml:.3f} {domain_3d[0] - pml:.3f} {1.25:.3f} {domain_3d[2] - pml:.3f} confined_material
#geometry_objects_read: {(domain_3d[0]/2 - self.x/2) :.3f} {domain_3d[1]/2 - self.y/2 - 0.25:.3f} {(domain_3d[2]/2 - self.z/2):.3f} {self.h5_file} Object_materials.txt
#geometry_view: 0 0 0 {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f} {self.resol} {self.resol} {self.resol} VoxelScan n

    '''     
            with open(self.input, 'w') as f:
                f.write(config)
                f.close()
            api(self.input, 
                n=1, 
                # gpu=[0],
                geometry_only=True, geometry_fixed=False)
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
    parser.add_argument('--num_scan', type=int, default=1, help='Number of A-Scans')

    args = parser.parse_args()
    rootimg = Roots_Func(args=args)
    rootimg.run_straight_scan()