#!/usr/bin/env python3
#####turn the antennas
import numpy as np
from gprMax.gprMax import api
from pathlib import Path
import os
# -----------------------------
# Scenario Configuration - OPTIMIZED FOR HYPERBOLAS
# -----------------------------
domain_x, domain_y, domain_z = 2.000, 1.100, 1.000
#domain_x, domain_y, domain_z = 1.000, 0.600, 0.500
dx_dy_dz = 0.01, 0.01, 0.01

center_x, center_y = domain_x / 2, 0.5
#circle_radius = 0.200  # Increased radius to create more distance variation
circle_radius = 0.2
num_steps = 120  # More steps for smoother hyperbola
antenna_offset = 0.05

output_dir = Path("circular_scan_results_air_large")
#output_dir.mkdir(exist_ok=True)
os.makedirs(output_dir)

src_coord = []
rx_coord = []

for i in range(num_steps):
#for i in range (66,120):
    filename = "circular_scan_results_air_large/circular_scan"+str(i)+".in"
    with open(filename, 'w') as f:
        # Write headers
        f.write(f"#title: Circular Scan Optimized for Hyperbolas\n")
        f.write(f"#domain: {domain_x} {domain_y} {domain_z}\n")
        f.write(f"#dx_dy_dz: {dx_dy_dz[0]} {dx_dy_dz[1]} {dx_dy_dz[2]}\n")
        f.write(f"#time_window: 60e-9\n\n")  # Increased time window

        # Define materials with higher contrast
        f.write("#material: 12.0 0.01 1.0 0.0 turang\n") #heterougeneous soil
        #f.write("#material: 1 1e8 1.0 0.0 metal\n")
        f.write("#material: 1.0 0 1.0 0.0 air\n\n")  

        # Create layered background
        #f.write(f"#box: 0 0 0 {domain_x} 1.000 {domain_z} turang\n")  # Soil
        f.write(f"#pml_cells: 20 20 20 20 20 20\n")
        f.write(f"#box: 0.2 0.2 0.2 1.8 0.9 0.8 air\n")
        #f.write(f"#box: 0.2 0.2 0.2 1.8 0.8 0.8 turang\n")  # Soil

        # CRITICAL: Add SHORT, OFF-CENTER horizontal cylinder
        pipe_radius = 0.200  # Smaller pipe
        pipe_depth = 0.08    # Burial depth
    
        # Pipe positioned away from center - this creates distance variation
        #pipe_x_start = center_x - 0.08
        #pipe_x_end = center_x + 0.08  
        #pipe_y = center_y + 0.12  # Positioned away from center in Y direction
        pipe_x_start = 0.1
        pipe_x_end = 0.8  
        pipe_y = 0.5  # Positioned away from center in Y direction
    
        #f.write(f"#cylinder: 1 0.5 0.2 1 0.5 0.8 {pipe_radius} pec\n\n")

        # Add a second pipe at different location for comparison
        pipe2_x_start = center_x - 0.1
        pipe2_x_end = center_x - 0.05  
        pipe2_y = center_y - 0.1
    
        #f.write(f"#cylinder: 1 0.5 0 1 0.5 1 0.19 air\n\n")

        # Define waveform
        f.write(f"#waveform: ricker 1 500e6 my_ricker\n\n")  # Lower frequency for deeper penetration

        # Circular scan geometry
        #antenna_height = 0.02
    
        angle = 2 * np.pi * i / num_steps
        
        # Tx position on circle
        src_x = center_x + circle_radius * np.cos(angle)
        src_y = 0.8
        src_z = center_y + circle_radius * np.sin(angle)
        
        # Rx position
        rx_x = center_x + (circle_radius - antenna_offset) * np.cos(angle)
        rx_y = 0.8
        rx_z = center_y + (circle_radius - antenna_offset) * np.sin(angle)
        '''
        with open('src_coord.txt', 'r') as file:
              for line in file:
                x, y, z = map(float,line.strip().split())
                src_coord.append((x, y, z))
        file.close()
        src_x, src_y, src_z = src_coord[i-1]

        with open('rx_coord.txt', 'r') as file:
            for line in file:
                x, y, z = map(float,line.strip().split())
                rx_coord.append((x, y, z))
        file.close()
        rx_x, rx_y, rx_z = rx_coord[i-1]
        '''

        f.write(f"#hertzian_dipole: z {src_x} {src_y} {src_z} my_ricker\n")
        f.write(f"#rx: {rx_x} {rx_y} {rx_z}\n\n")
        f.write(f"#geometry_view: 0 0 0 2.000 1.100 1.000 0.01 0.01 0.01 PEC n")
        f.close
    api(str(filename),n=1)

print(f"Input file created: {filename}")

# Run simulation
#api(str(filename), n=num_steps)
print("Simulation complete!")