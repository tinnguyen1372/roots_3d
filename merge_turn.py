import h5py
import numpy as np
import matplotlib.pyplot as plt
from tools.outputfiles_merge import merge_files
from tools.plot_Bscan import mpl_plot as mpl_plot_Bscan 

# 1. Merge the files
# Ensure the path matches where your circular scan outputs are saved
merge_files("./cir_scans/circular_scan", removefiles=False)
merged_file = "./cir_scans/circular_scan_merged.out"

def process_br(raw_data):
    """
    Apply Background Removal (Average Trace Subtraction).
    This is critical for circular scans to remove the strong 
    direct-coupling wave between Tx and Rx.
    """
    return raw_data - np.mean(raw_data, axis=1, keepdims=True)

# 2. Extract and Combine Data
with h5py.File(merged_file, 'r') as f:
    # Because we turned the dipole in the X-Z plane, 
    # the reflections will be across Ez and Ex.
    # Usually, for root detection, the sum of squares (Magnitude) 
    # or the dominant component is used.
    data_ez = f['rxs']['rx1']['Ez'][()]
    data_ex = f['rxs']['rx1']['Ex'][()]
    
    # Calculate Magnitude to capture the 'turned' field energy
    # This ensures you don't lose signal when the rotation passes 45 degrees
    data_combined = np.sqrt(data_ez**2 + data_ex**2)
    
    dt = f.attrs['dt']
    print(f"Data Shape: {data_combined.shape}")

# 3. Background Removal
data_processed = process_br(data_combined)

# 4. Visualization (B-Scan)
# We use the Magnitude to see the hyperbolas regardless of the antenna angle
mpl_plot_Bscan("Circular_Root_Scan_Magnitude", data_processed, dt, 1, 'Magnitude')

# 5. Polar Visualization (The "Top-Down" View)
# This maps the B-scan back onto a circle to see root locations
num_traces = data_processed.shape[1]
angles = np.linspace(0, 2 * np.pi, num_traces)
times = np.arange(data_processed.shape[0]) * dt

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
# Plotting a 'time-slice' or the whole radargram in polar coords
# We use a specific time range to avoid the initial direct wave pulse
time_start_idx = 500 # Adjust this to skip the initial 'blast' 
ax.pcolormesh(angles, times[time_start_idx:], data_processed[time_start_idx:], cmap='gray', shading='auto')
ax.set_yticklabels([]) # Hide time labels for a cleaner map
ax.set_title("Polar Mapping of Tree Roots")
plt.show()

# 6. Trace Inspection
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes.flatten()
for i in range(4):
    idx = i * (num_traces // 4)
    axes[i].plot(data_processed[:, idx])
    axes[i].set_title(f'Trace at {np.degrees(angles[idx]):.0f}°')
plt.tight_layout()
plt.show()