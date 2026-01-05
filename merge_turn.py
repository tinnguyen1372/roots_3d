import h5py
import numpy as np
import matplotlib.pyplot as plt
from tools.outputfiles_merge import merge_files
from tools.plot_Bscan import mpl_plot as mpl_plot_Bscan 

# 1. Merge output files from the circular scan
merge_files("./cir_scans/circular_scan", removefiles=False)
merged_file = "./cir_scans/circular_scan_merged.out"

def process_br(raw_data):
    """
    Applies Background Removal (Average Trace Subtraction).
    This removes the strong direct-coupling wave between Tx and Rx,
    which is essential for detecting targets like roots.
    """
    return raw_data - np.mean(raw_data, axis=1, keepdims=True)

# 2. Extract and combine horizontal field data
with h5py.File(merged_file, 'r') as f:
    # Capturing both horizontal components due to turned polarization
    data_ez = f['rxs']['rx1']['Ez'][()]
    data_ex = f['rxs']['rx1']['Ex'][()]
    
    # Magnitude captures the total horizontal energy regardless of antenna angle
    data_magnitude = np.sqrt(data_ez**2 + data_ex**2)
    
    dt = f.attrs['dt']
    print(f"Data Shape: {data_magnitude.shape}")

# 3. Apply Background Removal
data_processed = process_br(data_magnitude)

# 4. Standard B-Scan Visualization
# This should now show hyperbolas instead of solid blocks of color.
mpl_plot_Bscan("Turned_Polarization_BScan", data_processed, dt, 1, 'Magnitude')

# 5. Polar Visualization (Top-Down Mapping)
# This maps the B-scan data onto a circle to visualize root layout.
num_traces = data_processed.shape[1]
angles = np.linspace(0, 2 * np.pi, num_traces)
times = np.arange(data_processed.shape[0]) * dt

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
# time_start_idx skips the initial air wave 'blast' at the surface.
time_start_idx = 500 
ax.pcolormesh(angles, times[time_start_idx:], data_processed[time_start_idx:], cmap='gray', shading='auto')
ax.set_yticklabels([]) # For a cleaner root map
ax.set_title("Polar Mapping of Tree Roots")
plt.show()