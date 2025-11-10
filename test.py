import numpy as np

r = 1.5         # radius
delta = 0.005   # grid resolution
theta = np.linspace(0, 2*np.pi, 73)

# continuous circle
x = r * np.cos(theta)
y = r * np.sin(theta)

# quantized coordinates
xq = np.round(x / delta) * delta
yq = np.round(y / delta) * delta

# remove duplicates (optional)
points = np.unique(np.column_stack((xq, yq)), axis=0)

print(points.shape)

# INSERT_YOUR_CODE
np.savetxt('circle_points.txt', points, fmt='%.3f', delimiter=',', header='x,y')
