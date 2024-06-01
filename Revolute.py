import math
import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


log = False

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
(Might be good to use functools.partial)
'''
def parametric_revolution(fx, fy):
    fa = (lambda t, theta : fx(t) * math.cos(theta))
    fb = (lambda t, theta : fx(t) * math.sin(theta))
    fc = (lambda t, theta : fy(t))
    return fa, fb, fc


'''
Given a set of points in a 2D matrix, remove all duplicate points (with some tolerance) similar to math.isclose or numpy.isclose.
The default tolerance and keyword args are taken from numpy.isclose
pts has shape (len, num_pts)
return has shape (len, new_num_pts) for new_num_pts < num_pts
'''
def remove_duplicates(pts, rtol=1e-05, atol=1e-08):
    num_pts = pts.shape[1]
    # cdist assumes points are row-wise, with the form (num_pts, len)
    # our points are col-wise with form (len, num_pts)
    dist_matrix = distance.cdist(pts.T, pts.T)
    dist_matrix[np.arange(0, num_pts), np.arange(0, num_pts)] = math.inf
    min_dist = np.min(dist_matrix, axis = 0)
    to_discard = np.isclose(min_dist, 0, rtol=rtol, atol=atol)
    # to_keep = (to_discard == 0)
    to_keep = np.logical_not(to_discard)
    return pts[:, to_keep]
    
'''
Given a set of points, repeatedly apply an operation to them and return the result.
To make the algorithm more parallel, we split this into some rotation matrix, and a scaling factor. 
It might be useful to have the option to return a 3D matrix, where result[k, :, :] is the set of points after k applications

pts: (3,M) points to be transformed
op_rot: a scipy.spacial.transform.Rotation object
op_scale: a float denoting a scalar to multiply each successive application by
op_offset: a (3,1) array denoting an amount to add to each successive application. This offset is applied last. 
num_apps: an integer (N) denoting the maximum number of times the transform is applied. 
matrix_result: A boolean determining whether the result should be reshapen.

Return:
if matrix_result is False:
a (3, M * N) matrix
if matrix_result is True:
a (3, M, N) matrix, where the second axis (axis=1) denotes how many times the transform was applied. 
this means that result[:,k] is the (3,N) matrix of all the points after being transformed k times. 
'''
def repeated_transform(pts, op_rot : R, op_scale : float, op_offset, num_apps : int, matrix_result = False, log=False):
    num_pts = pts.shape[1]
    # Get the rotation as a rotation vector, since we can scale it to increase the rotation. 
    # has shape (N,)
    app_range = np.arange(0, num_apps+1)
    # has shape (3,)
    rot_vec = op_rot.as_rotvec()[np.newaxis, :]
    # The input for R.from_rotvec must be of the form (N, 3)
    # (N, 3) = (N, 1) @ (1, 3)
    each_rot_vec = app_range[:, np.newaxis] * rot_vec
    # all rotations
    all_rots = R.from_rotvec(each_rot_vec)
    # has shape (N, 3, 3)
    all_rot_mat = all_rots.as_matrix()
    # all scale; has shape (N,)
    all_scales = op_scale ** app_range
    # all offsets; has shape (3,N) 
    # (3, N) = (3, 1) @ (1, N)
    if(len(op_offset.shape) == 1):
        all_offsets = op_offset[:, np.newaxis] @ app_range[np.newaxis, :]
    elif(op_offset.shape[0] == 1):
        raise ValueError("offset vector has shape {0} when it must be in the form ({1},1)".format(op_offset.shape, np.max(op_offset.shape)))
    else:
        all_offsets = op_offset @ app_range[np.newaxis, :]
    
    # broadcasting of all three:
    # all_rots:    (N,3,3) will multiply (3, M) to get (N, 3, M) which we cyclically transpose into (3, M, N)
    if(log): print("M: {0} N: {1}".format(pts.shape[1], num_apps+1))
    aft_rot = all_rot_mat @ pts
    if(log): print("aft rotation shape:", aft_rot.shape)
    aft_rot = np.transpose(aft_rot, axes=(1,2,0))
    if(log): print("aft rotation transpose shape:", aft_rot.shape)
    # all_scales:  (N,) -> (M,N) -> (1,M,N) to broadcast to (3,M,N)
    all_scales_stretched = np.vstack(num_pts * (all_scales,))[np.newaxis, :, :]
    if(log): print("all scale stretch shape:", all_scales_stretched.shape)
    # all_offsets: (3,N) -> (3,M,N) repeated M times on axis=1
    all_offsets_stretched = np.stack( (num_pts) * (all_offsets,), axis=1)
    if(log): print("all offsets stretched shape:", all_offsets_stretched.shape)
    # The result after transforming all points should look like (3, M, N) 3D points, M points, N successive transformations
    full_result = aft_rot * all_scales_stretched + all_offsets_stretched
    if(matrix_result):
        return full_result
    else:
        return full_result.reshape(3, (num_apps + 1) * num_pts)
        
# Helper function to easily plot in 3D
def plot_3D(pts):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')     
    ax.scatter(pts[0], pts[1], pts[2])

# --- Some useful starting shapes ---
circle_theta    = np.linspace(0, 2 * math.pi, 50)
# r * (cos t, sin t) + (1, 0)
#basic_circle = 0.25 * np.vstack((np.cos(circle_theta), np.sin(circle_theta))) + np.array( len(circle_theta) * ([1,0],) ).T
#basic_circle = 0.25 * np.vstack((np.cos(circle_theta), np.sin(circle_theta))) + np.repeat( np.array([[1],[0]], len(circle_theta), axis = 1)
basic_circle_x = 0.25 * np.cos(circle_theta) + 1
basic_circle_y = 0.25 * np.sin(circle_theta)
basic_circle   = np.array((basic_circle_x, basic_circle_y))
basic_circle_3d = section_to_3D(basic_circle)
# Tilt the circle in 3D, to test 3D generatrix
circle_tilt = R.from_euler('x', math.pi / 4).as_matrix() @ basic_circle_3d

h_steps = np.linspace(-5, 5, 50)
h_line  = np.vstack((np.ones(50), h_steps, 2 * h_steps))
hyperboloid = surface_revolution(h_line)

# Generate a square with sidelength 0.3, centered at (1,0,0) parallel to the xz plane. 
sqr_pts = square_2D(0.3)
sqr = section_to_3D(sqr_pts)

helix_num = 50
dh_pts = np.array([[1,-1],[0,0],[0,0]])
dub_helix = repeated_transform(dh_pts, R.from_euler('z',math.pi/20), 1.0, np.array([0,0,0.1]), helix_num)

if( __name__ == '__main__'):
    # repeated transform test
    #rep_pts = repeated_transform(basic_circle_3d, R.from_euler('z', math.pi/8), 1.1, np.array([0,1,0]), 5)
    pass