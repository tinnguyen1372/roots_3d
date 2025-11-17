import numpy as np
r = 1         # radius
delta = 0.005   # grid resolution
theta = np.linspace(0, 2*np.pi, 72+1)

# continuous circle
cx = 3.200 / 2
cy = 3.200 / 2
x = cx + r * np.cos(theta)
y = cy + r * np.sin(theta)
# quantized coordinates
xq = np.round(x / delta) * delta
yq = np.round(y / delta) * delta

# Remove duplicates (preserve order) and sort clockwise
_, idx = np.unique(np.column_stack((xq, yq)), axis=0, return_index=True)
points = np.column_stack((xq, yq))[np.sort(idx)]
angles = np.arctan2(points[:,1] - cy, points[:,0] - cx)
points = np.round(points[np.argsort(angles)], 3)  # now clockwise
# INSERT_YOUR_CODE
with open('points.txt', 'w') as f:
    for pt in points:
        f.write(f"{pt[0]:.3f}, {pt[1]:.3f}\n")

for current_model_run in range(1, 10):
    # print(current_model_run)
    # print("waveform('gaussian', 1, 5e8, 'my_gaussian')")
    print(f"({points[current_model_run-1][0]}, {points[current_model_run-1][1]})")
    print(f"({points[current_model_run-2][0]},{points[current_model_run-2][1]:.3f})")

