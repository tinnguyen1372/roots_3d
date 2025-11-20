import numpy as np

input_number = int(input("Enter the number of model runs: "))
# current_model_run = int(input("Enter the current model run: "))
r_tx = 1         # radius
r_rx = 1.10
delta = 0.005   # grid resolution
theta = np.linspace(0, 2*np.pi, input_number+1)

# continuous circle
cx = 3.200 / 2
cy = 3.200 / 2
x_tx = cx + r_tx * np.cos(theta)
y_tx = cy + r_tx * np.sin(theta)

x_rx = cx + r_rx * np.cos(theta)
y_rx = cy + r_rx * np.sin(theta)
# quantized coordinates
xq_tx = np.round(x_tx / delta) * delta
yq_tx = np.round(y_tx / delta) * delta
xq_rx = np.round(x_rx / delta) * delta
yq_rx = np.round(y_rx / delta) * delta

_, idx = np.unique(np.column_stack((xq_tx, yq_tx)), axis=0, return_index=True)
_, idx_rx = np.unique(np.column_stack((xq_rx, yq_rx)), axis=0, return_index=True)
points_tx = np.column_stack((xq_tx, yq_tx))[np.sort(idx)]
points_rx = np.column_stack((xq_rx, yq_rx))[np.sort(idx)]
angles_tx = np.arctan2(points_tx[:,1] - cy, points_tx[:,0] - cx)
angles_rx = np.arctan2(points_rx[:,1] - cy, points_rx[:,0] - cx)
points_tx = np.round(points_tx[np.argsort(angles_tx)], 3)  
points_rx = np.round(points_rx[np.argsort(angles_rx)], 3)  

np.savetxt('points_tx.txt', points_tx, fmt='%.3f', delimiter=',')
np.savetxt('points_rx.txt', points_rx, fmt='%.3f', delimiter=',')