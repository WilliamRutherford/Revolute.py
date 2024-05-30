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
    return cylinder_coords[[1,2]]

'''
Given two cross sections which generate surfaces of revolution, compute the distance between them.  
The number in each need not be the same.
cross_section_a.shape = (2, n)
cross_section_b.shape = (2, m)
'''
def cross_section_dist(cross_section_a, cross_section_b):
    dist_matrix = distance.cdist(cross_section_a, cross_section_b)
    return np.sum(np.min(dist_matrix, axis = 0))

'''
Given two Generatrix sets of points (in 3D) which generate a surface of revolution, compute the distance between them.
The number in each need not be the same.
cross_section_a.shape = (3, n)
cross_section_b.shape = (3, m)
'''
def generatrix_dist(generatrix_a, generatrix_b):
    return cross_section_dist(cross_section(generatrix_a), cross_section(generatrix_b))

'''
Given two sources for a surface of revolution (cross section or generatrix) create a surface of resolution, and compute the distance between them. 
Why use this instead of the ones above? the process of revolution MIGHT introduce a multiplicative change.  
'''
def revolution_dist(source_a, source_b, angle_divs = 100):
    # If either is 2D (a cross section) convert to 3D. 
    if(len(source_a) == 2):
        pts_a = section_to_3D(source_a)
    else:
        pts_a = source_a
    if(len(source_b) == 2):
        pts_b = section_to_3D(source_b)
    else:
        pts_b = source_b
    rev_a = surface_revolution(pts_a, angle_divs = angle_divs)
    rev_b = surface_revolution(pts_b, angle_divs = angle_divs)
    dist_matrix = distance.cdist(rev_a, rev_b)
    return np.sum(np.min(dist_matrix, axis = 0))    

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
Given some parameters, generate a 2D square. 
side_len : the length of each side
center : the center point of the square in 2D space
'''
def square_2D(side_len, side_num = 10, center = np.array([1,0])):
    # The linearly spaced distances we will use to construct the sides
    line_pts = np.linspace(-side_len / 2, side_len / 2, side_num, endpoint = False)
    line_x = np.vstack((line_pts, np.zeros(side_num)))
    line_y = line_x[[1,0]]
    
    #center_x = np.array([center[0], 0        ])[:, np.newaxis]
    #center_y = np.array([0        , center[1]])[:, np.newaxis]
    
    side_x = np.array([side_len / 2, 0])[:, np.newaxis]
    side_y = side_x[[1,0]]
    
    if(log):
        print("line x:", line_x)
        print("line y:", line_y)
    l_side =  line_y - side_x
    r_side = -line_y + side_x
    u_side =  line_x + side_y
    d_side = -line_x - side_y
    
    if(log):
        print("sides shapes:")
        print(l_side.shape)
        print(r_side.shape)
        print(u_side.shape)
        print(d_side.shape)
    
    pts = np.concatenate((l_side, r_side, u_side, d_side), axis = 1)
    pts += center[:, np.newaxis]
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
Given a set of 2D points in the xy plane, twist them around the z axis. 
pts.shape = (2, m)
num_twists : the number of rotations per unit height
vert_divs: the number of vertical slices
z_bound: The upper and lower bound for the new twisted shape. 
'''
def twist(pts, num_twists = 1, vert_divs = 50, z_bound = (-1, 1)):
    z_offsets = np.linspace(z_bound[0], z_bound[1], vert_divs)
    angle_offsets = 2 * math.pi * num_twists * z_offsets
    # It will be easier if we convert the points to 3D, then convert to cylindrical and back
    # pts_cylindrical[0] += angle_offsets
    # pts_cylindrical[2] += z_offsets 

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

# Helper function to easily plot in 3D
def plot_3D(pts):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')     
    ax.scatter(pts[0], pts[1], pts[2])

# --- Some useful starting shapes ---
circle_theta    = np.linspace(0, 2 * math.pi, 50)
basic_circle    = np.vstack((0.25 * np.cos(circle_theta)+1, 0.25 * np.sin(circle_theta)))
basic_circle_3d = section_to_3D(basic_circle)
# Tilt the circle in 3D, to test 3D generatrix
circle_tilt = R.from_euler('x', math.pi / 4).as_matrix() @ basic_circle_3d

h_steps = np.linspace(-5, 5, 50)
h_line  = np.vstack((np.ones(50), h_steps, 2 * h_steps))
hyperboloid = surface_revolution(h_line)

# Generate a square with sidelength 0.3, centered at (1,0,0) parallel to the xz plane. 
sqr_pts = square_2D(0.3)
sqr = section_to_3D(sqr_pts)