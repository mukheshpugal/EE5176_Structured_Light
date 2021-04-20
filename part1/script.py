import open3d
import numpy as np
import cv2
from tqdm import tqdm

code = cv2.imread('./code.png', 0)
color = cv2.imread('./suzanne0000.png')

inverseTranslation = np.load('./invtrans.npy')
inverseRotation = np.load('./invrot.npy')
inverseProjection = np.transpose(np.load('./invprojx.npy'))

def unprojectCamera(nCam):
    pCam = inverseProjection @ nCam
    pCam = pCam / pCam[-1]

    return inverseTranslation @ inverseRotation @ pCam

def unprojectProjector(nPro):
    pPro = inverseProjection @ nPro
    pPro = pPro / pPro[-1]

    return pPro

def findCameraLine(u, v):
    nx = 2 * (u / 1920) - 1
    ny = 1 - 2 *  (v / 1080)

    pCam1 = unprojectCamera(np.array([nx, ny, .1, 1]))
    pCam2 = unprojectCamera(np.array([nx, ny, .7, 1]))

    return pCam1, pCam2 - pCam1

def findProjectorLine(projX):
    x = 2 * (projX / 1024) - 1.
    return unprojectProjector(np.array([x, 0, .7, 1]))

def intersect(cPos, cLine, pLine):
    def cross2D(a, b):
        return a[0] * b[2] - b[0] * a[2]

    p = np.array([0, 0, 0, 1.])
    r = pLine - p

    return cPos + cLine * (cross2D(p, r) - cross2D(cPos, r)) / cross2D(r, cLine)

ints = []
cols = []
for u in tqdm(range(1920)):
    for v in range(1080):
        cPos, cLine = findCameraLine(u, v)
        pLine = findProjectorLine(4 * code[v, u])
        intersection = intersect(cPos, cLine, pLine)
        ints.append(intersection)
        cols.append(color[v, u])
ints = np.stack(ints)
cols = np.stack(cols) / 255.

nn = np.linalg.norm(ints, axis=1)
ii = np.where(nn < 2)
ints = ints[ii]
cols = cols[ii]

def write():
    points = []
    for intt, coll in tqdm(zip(ints, cols)):
        points.append(f'{intt[0]} {intt[1]} {intt[2]/3} {coll[2]} {coll[1]} {coll[0]}\n')
    with open('./out.xyz', 'w') as f:
        f.write(f'{len(points)}\n')
        f.write('monke\n')
        f.writelines(points)
write()

# pcd = open3d.geometry.PointCloud()
# xyz = np.transpose(np.array([(xvals*zvals).reshape(-1), (yvals*zvals).reshape(-1), zvals.reshape(-1)]))
# pcd.points = open3d.utility.Vector3dVector(xyz)
# pcd.colors = open3d.utility.Vector3dVector(np.transpose(np.array([c.reshape(-1) for c in cv2.split(color)]))/255)

# open3d.visualization.draw_geometries([pcd])
