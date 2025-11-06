from tools.outputfiles_merge import merge_files
import h5py
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 
import numpy as np
merge_files("./Root3D/Roots/Roots", removefiles=False)
output_file = "./Root3D/Roots/Roots_merged.out"

merge_files("./Root3D/Base/Base", removefiles=False)
base_output_file = "./Root3D/Base/Base_merged.out"
def process_br(raw_ra):
    raw_br = raw_ra - np.mean(raw_ra, axis=1, keepdims=True)
    return raw_br

with h5py.File(output_file, 'r') as f1:
    print(f1.keys())
    data1 = f1['rxs']['rx1']['Ey'][()]
    dt = f1.attrs['dt']
    f1.close()

with h5py.File(base_output_file, 'r') as f1:
    print(f1.keys())
    data2 = f1['rxs']['rx1']['Ey'][()]
    dt = f1.attrs['dt']
    f1.close()

# data1 = data1[2000:,:]
# data1 = process_br(data1)
data = np.subtract(data1, data2)
rxnumber = 1
rxcomponent = 'Ey'
plt = mpl_plot_Bscan("merged_output_data", data, dt, rxnumber,rxcomponent)
fig_width = 15
fig_height = 15
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
plt.imshow(data, cmap='gray', aspect='auto')
plt.axis('off')
ax.margins(0, 0)  # Remove any extra margins or padding
fig.tight_layout(pad=0)  # Remove any extra padding
plt.show()