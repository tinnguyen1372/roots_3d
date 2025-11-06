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
from gprMax.gprMax import api

def generate_circle_xz(cx, cy, cz, radius, angle_step_deg):
    """
    Generate points on a circle in the XZ plane.
    
    cx, cy, cz : float
        Center of the circle
    radius : float
        Radius of the circle
    angle_step_deg : float
        Step size in degrees for generating points
    """
    points = []
    for angle_deg in np.arange(0, 360, angle_step_deg):
        theta = np.deg2rad(angle_deg)
        x = cx + radius * np.cos(theta)
        z = cz + radius * np.sin(theta)
        y = cy  # fixed for XZ plane
        points.append((x, y, z))
    return points

class Roots_Func():
    def __init__(self, args) -> None:
        # self.args = args
        # self.i = args.i
        # self.restart = 1
        # # self.num_scan = 20
        # self.num_scan = args.num_scan

        # self.resol = 0.005
        # self.time_window = 30e-9
        # self.gpu = args.gpu
        # self.square_size = args.square_size
        # self.air_thickness = args.air_thickness
        # self.confined_size = args.confined_size
        # # self.wall_height = args.wall_height
        # self.box_permittivity = args.box_permittivity
        # self.box_conductivity = args.box_conductivity
        # self.object_permittivity = args.object_permittivity
        # self.object_conductivity = args.object_conductivity
        # self.object_width = args.obj_width
        # self.object_height = args.obj_height
        self.filename = 'geometry_2d.h5'
        self.rootfile = 'root_2d.h5'
        self.num_scan = 36
        self.resol = 0.005
        self.time_window = 30e-9
        self.gpu = 0
        self.x = 1
        self.y = 1
        self.z = 0.5

        self.pix = int(max(self.x, self.y, self.z)/self.resol)
        self.confined_permittivity = 5.24
        self.confined_conductivity = 0.001
        self.object_permittivity = [24]
        self.object_conductivity = [0.0002]
        self.src_to_box = 0.1
        self.src_to_rx = 0.1
        self.confined_size = 20*0.005
        # self.fractal_box_seed = args.fractal_box_seed
        self.fractal_box_seed = random.randint(0,100)

        # Geometry load
        self.base = os.getcwd() + '/Geometry_ge/HeteSoil'
        self.basefile = self.base + '/hetesoil{}.png'.format(i)
        self.geofolder = os.getcwd() + '/Geometry_ge/Roots'
        self.geofile = self.geofolder + '/roots{}.png'.format(i)

        # Data load
        if not os.path.exists('./Input_ge'):
            os.makedirs('./Input_ge')        
        if not os.path.exists('./Input_ge/HeteSoil'):
            os.makedirs('./Input_ge/HeteSoil')
        if not os.path.exists('./Input_ge/ConfinedRoots'):
            os.makedirs('./Input_ge/ConfinedRoots')
        if not os.path.exists('./Input_ge/ConfinedHeteSoil'):
            os.makedirs('./Input_ge/ConfinedHeteSoil')
        if not os.path.exists('./Input_ge/Roots'):
            os.makedirs('./Input_ge/Roots')
        if not os.path.exists('./Output_ge'):
            os.makedirs('./Output_ge')
        if not os.path.exists('./Output_ge/HeteSoil'):
            os.makedirs('./Output_ge/HeteSoil')
        if not os.path.exists('./Output_ge/Roots'):
            os.makedirs('./Output_ge/Roots')
        if not os.path.exists('./Output_ge/SoilRoots'):
            os.makedirs('./Output_ge/SoilRoots')
        if not os.path.exists('./BaseImg_ge'):
            os.makedirs('./BaseImg_ge')
        if not os.path.exists('./ObjImg_ge'):
            os.makedirs('./ObjImg_ge')
        if not os.path.exists('./SoilRoots_ge'):
            os.makedirs('./SoilRoots_ge')
        if not os.path.exists('./Output_ge/ConfinedHeteSoil'):
            os.makedirs('./Output_ge/ConfinedHeteSoil')
        if not os.path.exists('./Output_ge/ConfinedRoots'):
            os.makedirs('./Output_ge/ConfinedRoots')
        if not os.path.exists('./ConfinedHeteSoil_ge'):
            os.makedirs('./ConfinedHeteSoil_ge')
        if not os.path.exists('./ConfinedRoots_ge'):
            os.makedirs('./ConfinedRoots_ge')
        


    def preprocess_3D(self, filename):
        img = Image.open(filename).convert('RGB')  # Convert the image to RGB mode

        color_map = {
            (255, 255, 255): -1,  # White (transparent)
            (255, 255, 0): -1,     # Yellow
            (255, 0, 0): 1,       # Red
            (0, 0, 255): 2,       # Blue
        }
        object_count = 2
        needed_size = object_count + 2
        color_map = dict(itertools.islice(color_map.items(), needed_size))

        def find_most_similar_color(pixel_color, color_map, threshold):
            closest_color = None
            for color, value in color_map.items():
                distance = np.linalg.norm(np.array(pixel_color) - np.array(color))
                if distance < threshold:
                    closest_color = color
            if closest_color is not None:
                return color_map[closest_color]
            else:
                return 0  # Return None when no similar color is found

        threshold = 100  # Adjust this threshold value as needed
        arr_2d = np.empty((self.pix, self.pix), dtype=int)
        arr_2d.fill(-1)
        arr_2d_root = np.empty((self.pix, self.pix), dtype=int)
        arr_2d_root.fill(-1)
        img_resized = img.resize((self.pix, self.pix))
        for y in range(self.pix):
            for x in range(self.pix):
                pixel_color = img_resized.getpixel((x, y))
                value = find_most_similar_color(pixel_color, color_map, threshold)
                arr_2d[y,x] = value
        arr_2d = np.rot90(arr_2d, k=-1)

        N = self.pix
        # XY plus
        # Combine to plus shape in XY
        arr_3d_x = np.tile(arr_2d[np.newaxis, :, :], (N, 1, 1))

        arr_3d_y = np.tile(arr_2d[:, np.newaxis, :], (1, N, 1))

        print("arr_3d_x shape:", arr_3d_x.shape)
        print("arr_3d_y shape:", arr_3d_y.shape)
        arr_3d = np.maximum(arr_3d_x, arr_3d_y)   # shape (N, N, N)

        # Rotate arr_3d by -90 degrees around the y-axis
        arr_3d = np.rot90(arr_3d, k=1, axes=(1, 2))

        # # Apply rotation
        # arr_3d = arr_3d @ R.T
        # print("arr_3d shape:", arr_3d.shape)

        # # Downsample
        # factor = 3
        # arr_3d_small = arr_3d[::factor, ::factor, ::factor]
        # print("Downsampled shape:", arr_3d_small.shape)

        # # Mask
        # mask = arr_3d_small != -1
        # print("Mask shape:", mask.shape)

        # # Plot
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.voxels(mask, facecolors='skyblue', edgecolor='k')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_xlim(0, 300)
        # ax.set_ylim(0, 300)
        # ax.set_zlim(0, 150)
        # ax.set_aspect('equal')
        # ax.set_title('3D Plus-Shaped Geometry')
        # plt.show()
        # arr_3d = np.expand_dims(arr_2d, axis=2)
        # arr_3d_root = np.expand_dims(arr_2d_root, axis=2)
        with h5py.File('./Geometry_3D/' + self.filename, 'w') as file:
            file.create_dataset("data", data=arr_3d)
            file.attrs['dx_dy_dz'] = (0.005, 0.005, 0.005)
            file.close()
    
    def generate_points_coord(self):
        self.cx = 1.6
        self.cy = 1.4
        self.cz = 1.6
        self.radius = 0.35
        self.angle_step = 360 / self.num_scan
        self.points = generate_circle_xz(self.cx, self.cy, self.cz, self.radius, self.angle_step)
        with open('points_coord.txt', 'w') as f:
            for point in self.points:
                f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n")
            f.close()
        return self.points

    def run_roots_3D(self):
        self.input = 'Roots.in'
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.05
        sharp_domain = 3 , 1.5, 3
        domain_3d = [
            float(sharp_domain[0] + 2 * pml), 
            float(sharp_domain[1] + 2 * pml), 
            float(sharp_domain[2] + 2 * pml)
        ]
        try:
            # with open('{}materials.txt'.format('Obj_'), "w") as file:
            #     file.write('#material: {} {} 1 0 box\n'.format(self.box_permittivity, self.box_conductivity))         
            self.preprocess_3D('geometry3.png')
            self.generate_points_coord()
        except Exception as e:
            print(f"Error in preprocess_3D and generate_points_coord:{e}")
        
        config =f'''
#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}
Source - Receiver - Waveform

#pml_cells: {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells}
#soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil
#material: 5.24 0.001 1 0 confined_material
#box: {pml:.3f} {1:.3f} {pml:.3f} {domain_3d[0] - pml:.3f} {1.15:.3f} {domain_3d[2] - pml:.3f} confined_material

#python:
cavity_coord = []
import numpy as np

cavity_coord = np.loadtxt('points_coord.txt')

cav_x, cav_y, cav_z = cavity_coord[current_model_run-1]
from user_libs.antennas.MALA_5mm import antenna_like_MALA_1200
antenna_like_MALA_1200(cav_x, cav_y, cav_z, resolution=0.005)
#end_python:

#fractal_box: {pml:.3f} {pml:.3f} {pml:.3f} {domain_3d[0] - pml:.3f} {1:.3f} {domain_3d[2] - pml:.3f} 1.5 1 1 1 20 hete_soil my_fractal_box {42}
#geometry_objects_read: {(domain_3d[0]/2 - self.x/2) :.3f} {domain_3d[1]/2 - self.z -0.25:.3f} {(domain_3d[2]/2 - self.y/2):.3f} Geometry_3D/geometry_2d.h5 Object_materials.txt
geometry_view: 0 0 0 {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f} 0.005 0.005 0.005 Base n

        '''
        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        # try:
        api(self.input, 
            n=self.num_scan, 
            # gpu=[3], 
            # restart=2,
            geometry_only=False, geometry_fixed=False)
        # except Exception as e:
        #     api(self.input, 
        #         n=1, 
        #         # gpu=[0], 
        #         # restart=self.restart,
        #         geometry_only=True, geometry_fixed=False)

    def run_base_3D(self):
        self.input = 'Base.in'
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.05
        sharp_domain = 3 , 1.5, 3
        domain_3d = [
            float(sharp_domain[0] + 2 * pml), 
            float(sharp_domain[1] + 2 * pml), 
            float(sharp_domain[2] + 2 * pml)
        ]
        try:
            # with open('{}materials.txt'.format('Obj_'), "w") as file:
            #     file.write('#material: {} {} 1 0 box\n'.format(self.box_permittivity, self.box_conductivity))         
            # self.preprocess_3D('geometry3.png')
            self.generate_points_coord()
        except Exception as e:
            print(f"Error in preprocess_3D and generate_points_coord:{e}")
        
        config =f'''
#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}
Source - Receiver - Waveform
#waveform: ricker 1 1e9 my_wave

#pml_cells: {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells}
#soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil
#material: 5.24 0.001 1 0 confined_material
#box: {pml:.3f} {1:.3f} {pml:.3f} {domain_3d[0] - pml:.3f} {1.15:.3f} {domain_3d[2] - pml:.3f} confined_material

#python:
cavity_coord = []
import numpy as np

cavity_coord = np.loadtxt('points_coord.txt')

cav_x, cav_y, cav_z = cavity_coord[current_model_run-1]
from user_libs.antennas.MALA_5mm import antenna_like_MALA_1200
antenna_like_MALA_1200(cav_x, cav_y, cav_z, resolution=0.005)
#end_python:
#fractal_box: {pml:.3f} {pml:.3f} {pml:.3f} {domain_3d[0] - pml:.3f} {1:.3f} {domain_3d[2] - pml:.3f} 1.5 1 1 1 20 hete_soil my_fractal_box {42}
geometry_view: 0 0 0 {domain_3d[0]:.3f} {domain_3d[1]:.3f} {domain_3d[2]:.3f} 0.005 0.005 0.005 Base n

        '''
        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        # try:
        api(self.input, 
            n=self.num_scan, 
            # gpu=[3], 
            # restart=2,
            geometry_only=False, geometry_fixed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roots Scanning for Through Imaging")      
    parser.add_argument('--start', type=int, default=0, help='Start of the generated geometry')
    parser.add_argument('--end', type=int, default=15, help='End of the generated geometry')
    parser.add_argument('--num_scan', type=int, default=100, help='Number of A-Scans')
    parser.add_argument('--gpu', type=int, default=0, help='Specify GPU')
    args = parser.parse_args()
    try:
        data = np.load('Geometry_3D/params_0_4999.npz', allow_pickle=True)
    except Exception as e:
        pass
    datasetvalue = 0

    for i in range(0, 1):
        i = i - datasetvalue
        # args.square_size = data['params'][i]['cube_size']/100
        # args.wall_thickness = data['params'][i]['wall_thickness']/100
        # args.wall_permittivity = round(data['params'][i]['permittivity_wall'], 2)
        # args.wall_conductivity = round(data['params'][i]['conductivity_wall'], 4)       
        # args.object_permittivity = [round(p, 2) for p in data['params'][i]['permittivity_object']]
        # args.object_conductivity = [round(p, 6) for p in data['params'][i]['conductivity_object']]
        args.i = i + datasetvalue

        rootimg = Roots_Func(args=args)
        # rootimg.run_base()
        # rootimg.run_2D()
        print(args)

        # rootimg.preprocess_3D('geometry2.png')

        # with open('points_coord.txt', 'w') as f:
            # points = rootimg.generate_points_coord()
        #     f.write(str(points))
        #     print(points[0])
        #     plt.scatter([p[0] for p in points], [p[2] for p in points])
        #     plt.xlabel('X')
        #     plt.ylabel('Z') 
        #     plt.title('Points Coordinates')
        #     plt.show()
        rootimg.run_base_3D()
        rootimg.run_roots_3D()