from tools.outputfiles_merge import merge_files
import h5py
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 
import numpy as np
merge_files("./straight_scans/straight_scan_1", removefiles=False)
merge_files("./straight_scans/straight_scan_2", removefiles=False)
merge_files("./straight_scans/straight_scan_3", removefiles=False)
merge_files("./straight_scans/straight_scan_4", removefiles=False)
output_files = ["./straight_scans/straight_scan_1_merged.out", "./straight_scans/straight_scan_2_merged.out", "./straight_scans/straight_scan_3_merged.out", "./straight_scans/straight_scan_4_merged.out"]
# merge_files("./Root3D/Base/Base", removefiles=False)
# base_output_file = "./Root3D/Base/Base_merged.out"
def process_br(raw_ra):
    raw_br = raw_ra - np.mean(raw_ra, axis=1, keepdims=True)
    return raw_br


data = None  # Initialize as None so we can handle the first file easily

for output_file in output_files:
    
    with h5py.File(output_file, 'r') as f1:
        data1 = f1['rxs']['rx1']['Ey'][()]
        print(data1.shape)
        dt = f1.attrs['dt']
    data1 = process_br(data1)
    if data is None:
        data = data1
    else:
        # data1 = np.subtract(data1, data[:,:20])
        data = np.concatenate((data, data1), axis=1)
# data1 = data1[2000:,:]
data = process_br(data)
# data = np.subtract(data1,)
# plt = mpl_plot_Bscan("merged_output_data", data, dt, rxnumber,rxcomponent)
import matplotlib.pyplot as plt
fig_width = 8
fig_height = 8
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
plt.imshow(data, cmap='gray', aspect='auto')
plt.axis('off')
ax.margins(0, 0)  # Remove any extra margins or padding
fig.tight_layout(pad=0)  # Remove any extra padding
plt.show()


import numpy as np
starting_point_tx = [(1,1.1,1), (1,1.1,2) , (2,1.1,2) , (2,1.1,1),(1,1.1,1)]
starting_point_rx = [(1,1.1,0.9), (0.9,1.1,2) , (2,1.1,2.1) , (2.1,1.1,1)]
tx = np.array(starting_point_tx)
rx = np.array(starting_point_rx)

plt.plot(tx[:, 0], tx[:, 2], 'ro-', label='Tx path')
plt.plot(rx[:, 0], rx[:, 2], 'b.', label='Rx path')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.legend()
plt.axis('equal')
plt.show()
