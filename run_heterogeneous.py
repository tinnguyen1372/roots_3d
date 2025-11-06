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



class HeteSoil_Func():
    def __init__(self, args) -> None:
        self.args = args
        self.i = args.i
        self.restart = 1
        # self.num_scan = 20
        self.num_scan = args.num_scan
        self.fractal_box_seed = args.fractal_box_seed

        self.resol = 0.005
        self.time_window = 30e-9
        self.gpu = args.gpu
        self.square_size = args.square_size
        self.air_thickness = args.air_thickness
        self.confined_size = args.confined_size
        # self.wall_height = args.wall_height
        self.box_permittivity = args.box_permittivity
        self.box_conductivity = args.box_conductivity
        self.object_permittivity = args.object_permittivity
        self.object_conductivity = args.object_conductivity
        # self.object_width = args.obj_width
        # self.object_height = args.obj_height
        self.src_to_box = 0.1
        self.src_to_rx = 0.1
        # Geometry load
        self.base = os.getcwd() + '/Geometry_ge/HeteSoil'
        self.basefile = self.base + '/hetesoil{}.png'.format(i)
        self.geofolder = os.getcwd() + '/Geometry_ge/Roots'
        self.geofile = self.geofolder + '/roots{}.png'.format(i)

        # Data load
        self.pix =int(self.square_size/0.005)
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
        

  
    def view_geometry(self):
        # self.preprocess(self.basefile)
        with h5py.File('./Geometry_ge/geometry_2d.h5', 'r') as f:
            data = f['data'][:]
        
        # Adjust large_array to match data's shape
        data = np.squeeze(data, axis=2)  # Remove any singleton dimensions, if needed
        large_array = np.full(data.shape, -1, dtype=int)
        # Override the values in large_array with data
        large_array[:data.shape[0], :data.shape[1]] = data

        # Mask the regions where the value is 1
        masked_data = ma.masked_where(large_array == -1, large_array)

        # Marker positions based on provided coordinates and scaling factor
        # marker_x, marker_y = 0.15 * data.shape[0] /3.33, 0.15 * data.shape[0] /3.33
        color_list = [
            (1.0, 1.0, 1.0),  # White for -1
            (1.0, 1.0, 0.0),  # Yellow for 0
            (1.0, 0.0, 0.0),   # Red for 1
            (0.0, 0.0, 1.0)
        ]
        custom_cmap = ListedColormap(color_list, name="custom_cmap")
        # Plot the markers and masked data
        # plt.plot(marker_x, marker_y, marker='o', color='red', markersize=5)
        plt.imshow(masked_data, cmap='viridis')
        plt.axis('equal')
        plt.title("Geometry Visualization")
        plt.xlabel("X-axis (pixels)")
        plt.ylabel("Y-axis (pixels)")
        plt.show()

    def preprocess(self, filename):
        from PIL import Image
        import numpy as np
        import h5py

        img = Image.open(filename).convert('RGB')  # Convert the image to RGB mode
        # img.show()
        # print(self.pix)
        # Define the color map with a tolerance

        # Base color map
        color_map = {
            (255, 255, 255): -1,  # White (transparent)
            (255, 255, 0): 0,     # Yellow
            (0, 0, 255): 3,       # Blue
        }

        # Limit the dictionary to needed size
        needed_size = len(self.object_permittivity) + 2
        color_map = dict(itertools.islice(color_map.items(), needed_size))

        # Print for debuggings
        # print(color_map)

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
        # Define the threshold
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
                if value == 3:
                    arr_2d_root[y,x] = value - 3
                if value == 0:
                    arr_2d[y,x] = value
                # arr_2d[y, x] = find_most_similar_color(pixel_color, color_map, threshold)
        arr_2d = np.rot90(arr_2d, k=-1)
        arr_2d_root = np.rot90(arr_2d_root, k=-1)
        # np.savetxt('output_array.txt', arr_2d, fmt='%d', delimiter=' ')
        self.filename = 'geometry_2d.h5'
        self.rootfile = 'root_2d.h5'
        arr_3d = np.expand_dims(arr_2d, axis=2)
        arr_3d_root = np.expand_dims(arr_2d_root, axis=2)
        with h5py.File('./Geometry_ge/' + self.filename, 'w') as file:
            file.create_dataset("data", data=arr_3d)
            file.attrs['dx_dy_dz'] = (0.005, 0.005, 0.005) 
            file.close()       
        with h5py.File('./Geometry_ge/' + self.rootfile, 'w') as file:
            file.create_dataset("data", data=arr_3d_root)
            file.attrs['dx_dy_dz'] = (0.005, 0.005, 0.005)
            file.close()

    def run_base(self):

        # Run gprMax
        self.input = './Input_ge/HeteSoil/HeteSoil{}.in'.format(self.i)
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.05

        sharp_domain = self.square_size, self.square_size
        domain_2d = [
            float(sharp_domain[0] + 2 * pml), 
            float(sharp_domain[1] + 2 * pml), 
            0.005
        ]

        # Preprocess geometry
        try:
            with open('{}materials.txt'.format('Obj_'), "w") as file:
                file.write('#material: {} {} 1 0 box\n'.format(self.box_permittivity, self.box_conductivity))         
            self.preprocess(self.geofile)
        except Exception as e:
            print(e)

        src_position = [src_to_pml + pml, 
                        self.square_size - self.air_thickness + self.src_to_box + pml + self.confined_size, 
                        0]
        rx_position = [src_to_pml + pml + self.src_to_rx, 
                       self.square_size - self.air_thickness + self.src_to_box + pml + self.confined_size, 
                       0]        
        
        src_steps = [(self.square_size)/ self.num_scan, 0, 0]
        config = f'''

#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0

Source - Receiver - Waveform
#waveform: ricker 1 6e8 my_wave

#soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil

#hertzian_dipole: z {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/geometry_2d.h5 Obj_materials.txt
#fractal_box: {pml:.3f} {pml:.3f} 0 {domain_2d[0] - pml:.3f} {self.square_size-self.air_thickness+pml:.3f} 0.005 1.5 1 1 1 20 hete_soil my_fractal_box {self.fractal_box_seed}
geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/root_2d.h5 Root_materials.txt
#geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Roots 
#geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 HeteSoil{self.i} n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                # n=self.num_scan - self.restart - 9,
                # gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                # n=self.num_scan - self.restart - 9,  
                # gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        try:
        
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            uncleaned_output_file = f'./Output_ge/HeteSoil/HeteSoil{self.i}.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ez'][()]
                dt = f1.attrs['dt']
                f1.close()

            # with h5py.File(f'./Output_ge/Base/Base{self.i}.out', 'r') as f1:
            #     data_source = f1['rxs']['rx1']['Ez'][()]

            with h5py.File(uncleaned_output_file, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ez', data=data1)
                f_out.close()
            rxnumber = 1
            rxcomponent = 'Ez'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            fig_width = 15
            fig_height = 15
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding
            plt.savefig(f'./HeteSoil_ge/HeteSoil{self.i}' + ".png")
            plt.close()
            # data1 = np.subtract(data1, data_source)

            # with h5py.File(output_file, 'w') as f_out:
            #     f_out.attrs['dt'] = dt  # Set the time step attribute
            #     f_out.create_dataset('rxs/rx1/Ez', data=data1)

            # # Draw data with normal plot
            # rxnumber = 1
            # rxcomponent = 'Ez'
            # plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            # fig_width = 15
            # fig_height = 15

            # fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # plt.imshow(data1, cmap='gray', aspect='auto')
            # plt.axis('off')
            # ax.margins(0, 0)  # Remove any extra margins or padding
            # fig.tight_layout(pad=0)  # Remove any extra padding

            # os.rename(output_file, f'./Output_ge/Object/Obj{self.i}.out')
            # plt.savefig(f'./ObjImg_ge/Obj{self.i}' + ".png")
        except Exception as e:
            print(e)


    def run_2D(self):

        # Run gprMax
        self.input = './Input_ge/Roots/Roots{}.in'.format(self.i)
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.05

        sharp_domain = self.square_size, self.square_size
        domain_2d = [
            float(sharp_domain[0] + 2 * pml), 
            float(sharp_domain[1] + 2 * pml), 
            0.005
        ]


        # Preprocess geometry

        try:
            with open('{}materials.txt'.format('Obj_'), "w") as file:
                file.write('#material: {} {} 1 0 box\n'.format(self.box_permittivity, self.box_conductivity))
            with open('{}materials.txt'.format('Root_'), "w") as file:
                for i in range(len(self.object_permittivity)):
                    file.write('#material: {} {} 1 0 Object{}\n'.format(self.object_permittivity[i],self.object_conductivity[i],i))          
                self.preprocess(self.geofile)
        except Exception as e:
            print(e)

        src_position = [src_to_pml + pml, 
                        self.square_size - self.air_thickness + self.src_to_box + pml + self.confined_size, 
                        0]
        rx_position = [src_to_pml + pml + self.src_to_rx, 
                       self.square_size - self.air_thickness + self.src_to_box + pml + self.confined_size, 
                       0]        
        
        src_steps = [(self.square_size)/ self.num_scan, 0, 0]
        config = f'''

#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0

Source - Receiver - Waveform
#waveform: ricker 1 6e8 my_wave

#soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil

#hertzian_dipole: z {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/geometry_2d.h5 Obj_materials.txt
#fractal_box: {pml:.3f} {pml:.3f} 0 {domain_2d[0] - pml:.3f} {self.square_size-self.air_thickness+pml:.3f} 0.005 1.5 1 1 1 20 hete_soil my_fractal_box {self.fractal_box_seed}
#geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/root_2d.h5 Root_materials.txt
#geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Roots 
#geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 HeteRoot{self.i} n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                # n=self.num_scan - self.restart - 9,
                # gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                # n=self.num_scan - self.restart - 9,  
                # gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        try:
        
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            uncleaned_output_file = f'./Output_ge/SoilRoots/SoilRoots{self.i}.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ez'][()]
                dt = f1.attrs['dt']
                f1.close()

            # with h5py.File(f'./Output_ge/Base/Base{self.i}.out', 'r') as f1:
            #     data_source = f1['rxs']['rx1']['Ez'][()]

            with h5py.File(uncleaned_output_file, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ez', data=data1)
                f_out.close()
            rxnumber = 1
            rxcomponent = 'Ez'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            fig_width = 15
            fig_height = 15
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding
            plt.savefig(f'./SoilRoots_ge/SoilRoots{self.i}' + ".png")
            plt.close()
            # data1 = np.subtract(data1, data_source)

            # with h5py.File(output_file, 'w') as f_out:
            #     f_out.attrs['dt'] = dt  # Set the time step attribute
            #     f_out.create_dataset('rxs/rx1/Ez', data=data1)

            # # Draw data with normal plot
            # rxnumber = 1
            # rxcomponent = 'Ez'
            # plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            # fig_width = 15
            # fig_height = 15

            # fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # plt.imshow(data1, cmap='gray', aspect='auto')
            # plt.axis('off')
            # ax.margins(0, 0)  # Remove any extra margins or padding
            # fig.tight_layout(pad=0)  # Remove any extra padding

            # os.rename(output_file, f'./Output_ge/Object/Obj{self.i}.out')
            # plt.savefig(f'./ObjImg_ge/Obj{self.i}' + ".png")
        except Exception as e:
            print(e)
### ------------------------------------------------------------------CONFINED ENVIRONMENT--------------------------------------------
    def run_confined_base(self):

        # Run gprMax
        self.input = './Input_ge/ConfinedHeteSoil/ConfinedHeteSoil{}.in'.format(self.i)
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.05

        sharp_domain = self.square_size, self.square_size
        domain_2d = [
            float(sharp_domain[0] + 2 * pml), 
            float(sharp_domain[1] + 2 * pml), 
            0.005
        ]

        # Preprocess geometry
        try:
            with open('{}materials.txt'.format('Obj_'), "w") as file:
                file.write('#material: {} {} 1 0 box\n'.format(self.box_permittivity, self.box_conductivity))         
            self.preprocess(self.geofile)
        except Exception as e:
            print(e)

        src_position = [src_to_pml + pml, 
                        self.square_size - self.air_thickness + self.src_to_box+ self.confined_size + pml, 
                        0]
        rx_position = [src_to_pml + pml + self.src_to_rx, 
                       self.square_size - self.air_thickness + self.src_to_box + self.confined_size + pml, 
                       0]      
        
        src_steps = [(self.square_size)/ self.num_scan, 0, 0]
        config = f'''

#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0

Source - Receiver - Waveform
#waveform: ricker 1 6e8 my_wave

#material: 5.24 0.001 1 0 confined_material
#soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil

#hertzian_dipole: z {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/geometry_2d.h5 Obj_materials.txt
#fractal_box: {pml:.3f} {pml:.3f} 0 {domain_2d[0] - pml:.3f} {self.square_size-self.air_thickness+pml:.3f} 0.005 1.5 1 1 1 20 hete_soil my_fractal_box {self.fractal_box_seed}
geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/root_2d.h5 Root_materials.txt
#geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Roots
#box: {pml:.3f} {self.square_size-self.air_thickness+pml:.3f} {0:.3f} {domain_2d[0] - pml:.3f} {self.square_size-self.air_thickness+self.confined_size+pml:.3f} 0.005 confined_material
#geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 ConfinedHeteSoil{self.i} n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                # n=self.num_scan - self.restart - 9,
                # gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                # n=self.num_scan - self.restart - 9,  
                # gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        try:
        
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            uncleaned_output_file = f'./Output_ge/ConfinedHeteSoil/ConfinedHeteSoil{self.i}.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ez'][()]
                dt = f1.attrs['dt']
                f1.close()

            # with h5py.File(f'./Output_ge/Base/Base{self.i}.out', 'r') as f1:
            #     data_source = f1['rxs']['rx1']['Ez'][()]

            with h5py.File(uncleaned_output_file, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ez', data=data1)
                f_out.close()
            rxnumber = 1
            rxcomponent = 'Ez'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            fig_width = 15
            fig_height = 15
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding
            plt.savefig(f'./COnfinedHeteSoil_ge/ConfinedHeteSoil{self.i}' + ".png")
            plt.close()
            # data1 = np.subtract(data1, data_source)

            # with h5py.File(output_file, 'w') as f_out:
            #     f_out.attrs['dt'] = dt  # Set the time step attribute
            #     f_out.create_dataset('rxs/rx1/Ez', data=data1)

            # # Draw data with normal plot
            # rxnumber = 1
            # rxcomponent = 'Ez'
            # plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            # fig_width = 15
            # fig_height = 15

            # fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # plt.imshow(data1, cmap='gray', aspect='auto')
            # plt.axis('off')
            # ax.margins(0, 0)  # Remove any extra margins or padding
            # fig.tight_layout(pad=0)  # Remove any extra padding

            # os.rename(output_file, f'./Output_ge/Object/Obj{self.i}.out')
            # plt.savefig(f'./ObjImg_ge/Obj{self.i}' + ".png")
        except Exception as e:
            print(e)


    def run_confined2D(self):

        # Run gprMax
        self.input = './Input_ge/ConfinedRoots/ConfinedRoots{}.in'.format(self.i)
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.05

        sharp_domain = self.square_size, self.square_size
        domain_2d = [
            float(sharp_domain[0] + 2 * pml), 
            float(sharp_domain[1] + 2 * pml), 
            0.005
        ]


        # Preprocess geometry

        try:
            with open('{}materials.txt'.format('Obj_'), "w") as file:
                file.write('#material: {} {} 1 0 box\n'.format(self.box_permittivity, self.box_conductivity))
            with open('{}materials.txt'.format('Root_'), "w") as file:
                for i in range(len(self.object_permittivity)):
                    file.write('#material: {} {} 1 0 Object{}\n'.format(self.object_permittivity[i],self.object_conductivity[i],i))          
                self.preprocess(self.geofile)
        except Exception as e:
            print(e)

        src_position = [src_to_pml + pml, 
                        self.square_size - self.air_thickness + self.src_to_box + self.confined_size + pml, 
                        0]
        rx_position = [src_to_pml + pml + self.src_to_rx, 
                       self.square_size - self.air_thickness + self.src_to_box + self.confined_size + pml, 
                       0]      
        
        src_steps = [(self.square_size)/ self.num_scan, 0, 0]
        config = f'''

#title: Roots under Hete Soil Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} 0 {pml_cells} {pml_cells} 0

Source - Receiver - Waveform
#waveform: ricker 1 6e8 my_wave

#material: 5.24 0.001 1 0 confined_material
#soil_peplinski: 0.3 0.7 2 2.66 0.01 0.15 hete_soil

#hertzian_dipole: z {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/geometry_2d.h5 Obj_materials.txt
#fractal_box: {pml:.3f} {pml:.3f} 0 {domain_2d[0] - pml:.3f} {self.square_size-self.air_thickness+pml:.3f} 0.005 1.5 1 1 1 20 hete_soil my_fractal_box {self.fractal_box_seed}
#geometry_objects_read: {pml:.3f} {pml:.3f} {0:.3f} Geometry_ge/root_2d.h5 Root_materials.txt
#geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Roots
#box: {pml:.3f} {self.square_size-self.air_thickness+pml:.3f} {0:.3f} {domain_2d[0] - pml:.3f} {self.square_size-self.air_thickness+self.confined_size+pml:.3f} 0.005 confined_material
#geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 ConfinedRoot{self.i} n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                #n=self.num_scan - self.restart - 9,
                #gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                #n=self.num_scan - self.restart - 9,  
                #gpu=[self.gpu],
                restart=self.restart,
                geometry_only=True, geometry_fixed=False)
        try:
        
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            uncleaned_output_file = f'./Output_ge/ConfinedRoots/ConfinedRoots{self.i}.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ez'][()]
                dt = f1.attrs['dt']
                f1.close()

            # with h5py.File(f'./Output_ge/Base/Base{self.i}.out', 'r') as f1:
            #     data_source = f1['rxs']['rx1']['Ez'][()]

            with h5py.File(uncleaned_output_file, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ez', data=data1)
                f_out.close()
            rxnumber = 1
            rxcomponent = 'Ez'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            fig_width = 15
            fig_height = 15
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding
            plt.savefig(f'./ConfinedRoots/ConfinedRoots{self.i}' + ".png")
            plt.close()
            # data1 = np.subtract(data1, data_source)

            # with h5py.File(output_file, 'w') as f_out:
            #     f_out.attrs['dt'] = dt  # Set the time step attribute
            #     f_out.create_dataset('rxs/rx1/Ez', data=data1)

            # # Draw data with normal plot
            # rxnumber = 1
            # rxcomponent = 'Ez'
            # plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            # fig_width = 15
            # fig_height = 15

            # fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # plt.imshow(data1, cmap='gray', aspect='auto')
            # plt.axis('off')
            # ax.margins(0, 0)  # Remove any extra margins or padding
            # fig.tight_layout(pad=0)  # Remove any extra padding

            # os.rename(output_file, f'./Output_ge/Object/Obj{self.i}.out')
            # plt.savefig(f'./ObjImg_ge/Obj{self.i}' + ".png")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heterogenous Soil Scanning for Through Imaging")      
    parser.add_argument('--start', type=int, default=0, help='Start of the generated geometry')
    parser.add_argument('--end', type=int, default=15, help='End of the generated geometry')
    parser.add_argument('--num_scan', type=int, default=100, help='Number of A-Scans')
    parser.add_argument('--gpu', type=int, default=3, help='Specify GPU')
    # data = np.load('SL_Objgeall_0_699.npz', allow_pickle=True)
    # data = np.load('SL_Objgeall_700_1500.npz', allow_pickle=True)
    # data = np.load('Geometry_ge/4w_multi_0_999.npz', allow_pickle=True)
    # data = np.load('Geometry_ge/4w_multi_1000_1999.npz', allow_pickle=True)
    # data = np.load('Geometry_ge/4w_multi_0_4999.npz', allow_pickle=True)
    # data = np.load('Geometry_ge/4w_multi_5000_9999.npz', allow_pickle=True)
    data = np.load('Geometry_ge/root_hete_0_499.npz', allow_pickle=True)
    datasetvalue = 0
    confined_size = 20*0.005
    fractal_box_seed = random.randint(0,100)
    # fractal_box_seed = 19
    args = parser.parse_args()
    for i in range(args.start, args.end):
        i = i - datasetvalue
        #print(data['params'][i]['square_size'])
        args.square_size = data['params'][i]['square_size']/100
        args.air_thickness = data['params'][i]['air_thickness']/100
        args.box_permittivity = round(data['params'][i]['permittivity_box'], 3)
        args.box_conductivity = round(data['params'][i]['conductivity_box'], 6)
        args.object_permittivity = [round(p, 3) for p in data['params'][i]['permittivity_object']]
        args.object_conductivity = [round(p, 6) for p in data['params'][i]['conductivity_object']]
        args.fractal_box_seed = fractal_box_seed
        args.confined_size = confined_size
        args.i = i + datasetvalue
    # start  adaptor
        heteImg = HeteSoil_Func(args=args)
        # wallimg.view_geometry()
        heteImg.run_base()
        heteImg.run_2D()
        heteImg.run_confined_base()
        heteImg.run_confined2D()
