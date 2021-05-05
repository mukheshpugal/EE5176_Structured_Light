import cv2
import numpy as np
from tqdm import tqdm

K = np.array([[567.6, 0, 320], [0, 570.2, 240], [0 ,0, 1]], dtype=np.float32)
disp = np.load('dispx.npy')
disp[np.where(disp < 204.62)] = 0
f = 570
depth = f * 0.075 / (disp + 1e-3)
ambient = np.load('ambient0_0.npy')[0]

invK = np.linalg.inv(K)

# x = cv2.reprojectImageTo3D(disp, Q)
x = np.zeros((480, 640, 3))
for u in tqdm(range(640)):
    for v in range(480):
        px = u - 320
        py = v - 240
        k = (f**2 + px**2 + py**2)**0.5
        d = depth[v, u]
        x[v, u] = np.array([px*d/k, py*d/k, f*d/k])

points = [f'{location[0]} {location[1]} {location[2]} {color} {color} {color}\n' for location, color in tqdm(list(zip(x.reshape(-1, 3), ambient.reshape(-1)))) if location[2] < 5]
with open('./out.xyz', 'w') as f:
    f.write(f'{len(points)}\n')
    f.write('chair\n')
    f.writelines(points)
