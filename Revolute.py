import math
import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


log = True

'''
Given a set of m 3d points, convert them to cylindrical coordinates. 
(x, y, z) -> (theta, r, z)

the "theta" is within the xy plane, starting from (1, 0, 0)

pts.shape = (3, m)

-math.pi <= theta[i] <= math.pi
'''
def cylindrical(pts):
    Z = pts[2]
    R = np.linalg.norm(pts[[0,1]], axis = 0)
    theta = np.arctan2(pts[1], pts[0])
    return np.vstack((theta, R, Z))

'''
Given a set of points in cylindrical coordinates, convert them to 3D. 
(theta, r, z) -> (x, y, z)
'''
def from_cylindrical(cyld_pts):
    X = cyld_pts[1] * np.cos(cyld_pts[0])
    Y = cyld_pts[1] * np.sin(cyld_pts[0])
    Z = cyld_pts[2]
    return np.vstack((X, Y, Z))

'''
Given 3D points which represents a generatrix, convert it to 2D points which generates an equivalent surface of revolution.
If we *do* use this cross section to generate a surface of revolution, the distribution of points will be completely different. 
The points given represent the xz plane. 
'''
def cross_section(generatrix_pts):
    cylinder_coords = cylindrical(generatrix_pts)
    # Remove the angle theta, so they are all pushed to the xz plane, which we now consider the xy plane. 
    return cylinder_coords[:, (1,2)]

'''
Given two cross sections which generate surfaces of revolution, compute the distance between them.  
The number in each need not be the same.
cross_section_a.shape = (n, 2)
cross_section_b.shape = (m, 2)
'''
def cross_section_dist(cross_section_a, cross_section_b):
    dist_matrix = distance.cdist(cross_section_a, cross_section_b)
    return np.sum(np.min(dist_matrix, axis = 0))

def generatrix_dist(generatrix_a, generatrix_b):
    return cross_section_dist(cross_section(generatrix_a), cross_section(generatrix_b))

'''
Given a set of 2D points, convert them to 3D. they are represented in the xz plane. 
(a, b) -> (a, 0, b)
'''
def section_to_3D(pts_2d):
    pts = pts_2d.copy()
    # (a, b) -> (a, b, 0)
    pts.resize(3, pts.shape[1])
    # (a, b, 0) -> (a, 0, b)
    pts = pts[[0, 2, 1], :]
    return pts


'''
Given a set of points (2D or 3D) revolve them around the Z axis and return a new set of points. 

If angle_divs is given, the same shape will be repeated at 'angle_divs' linearly spaced angles.
If relative_divs is given, it represents the number of points around the revolution we would generate for any point with radius = 1. 
    Every other point will generate a circle of points with approximately the same linear spacing. 
    This ensures more equally distributed revolved points, but loses all structure of the canonical points. 
'''
def surface_revolution(canonical_pts, angle_divs = 60, relative_divs = None, log = False):
    # If canonical_pts is 2D, convert it into 3D on the xz plane. 
    if(canonical_pts.shape[0] == 2):
        pts = section_to_3D(canonical_pts)
        
    else:
        pts = canonical_pts
        
    if(relative_divs is None):
        theta = np.linspace(0, 2 * math.pi, angle_divs, endpoint=False)
        all_rots = R.from_euler('z', theta).as_matrix()
        if(log):
            print("all rotations shape:", all_rots.shape)
            print("canonical pts shape:", pts.shape)
        all_pts = all_rots @ pts
        all_pts = np.transpose(all_pts, (1, 0, 2))
    
        
        if(log):
            print("all points shape:", all_pts.shape)
        return all_pts.reshape(3, angle_divs * pts.shape[1])
    else:
        raise NotImplementedError("for Surface Revolution, use 'angle_divs' instead of 'relative_divs'")


'''
Given a parametric function described by x(t) and y(t), return a parametric shape of revolution. 

first, we convert to cylindrical coordinates: 
r(theta) = x(t)
theta = theta

Then we convert to 3D. 
a(t) = r(theta) * cos(theta) = x(t) * cos(theta)
b(t) = r(theta) * sin(theta) = x(t) * sin(theta)
c(t) = y(t)

The result will be three functions; a(t, theta), b(t, theta), c(t, theta)
'''
def parametric_revolution(fx, fy):
    fa = (lambda t, theta : fx(t) * math.cos(theta))
    fb = (lambda t, theta : fx(t) * math.sin(theta))
    fc = (lambda t, theta : fy(t))
    return fa, fb, fc
    
circle_theta    = np.linspace(0, 2 * math.pi, 50)
basic_circle    = np.vstack((0.25 * np.cos(circle_theta)+1, 0.25 * np.sin(circle_theta)))
basic_circle_3d = section_to_3D(basic_circle)

# Tilt the circle in 3D, to test 3D generatrix
circle_tilt = R.from_euler('x', math.pi / 4).as_matrix() @ basic_circle_3d

# Helper function to easily plot in 3D
def plot_3D(pts):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')     
    ax.scatter(pts[0], pts[1], pts[2])