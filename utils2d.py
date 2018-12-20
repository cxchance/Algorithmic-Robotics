import numpy
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def view_pc(pcs, fig=None, color='b', marker='o'):
    """Visualize a pc.

    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        color - specifies the color of each point cloud.
            if a single value all point clouds will have that color.
            if an array of the same length as pcs each pc will be the color corresponding to the
            element of color.
        marker - specifies the marker of each point cloud.
            if a single value all point clouds will have that marker.
            if an array of the same length as pcs each pc will be the marker corresponding to the
            element of marker.
    outputs:
        fig - the pyplot figure that the point clouds are plotted on

    """
    # Construct the color and marker arrays
    if hasattr(color, '__iter__'):
        if len(color) != len(pcs):
            raise Exception('color is not the same length as pcs')
    else:
        color = [color] * len(pcs)

    if hasattr(marker, '__iter__'):
        if len(marker) != len(pcs):
            raise Exception('marker is not the same length as pcs')
    else:
        marker = [marker] * len(pcs)

    # Start plt in interactive mode
    ax = []
    if fig == None:
        plt.ion()
        # Make a 3D figure
        fig = plt.figure()

    # Draw each point cloud
    for pc, c, m in zip(pcs, color, marker):
        x = []
        y = []
        for pt in pc:
            x.append(pt[0, 0])
            y.append(pt[1, 0])

        plt.scatter(x, y, color=c, marker=m)

    # Set the labels
    plt.xlabel('X')
    plt.ylabel('Y')
    # Update the figure
    plt.show()

    # Return a handle to the figure so the user can make adjustments
    return fig


def add_noise(pc, variance,distribution='gaussian'):
    """Add Gaussian noise to pc.

    For each dimension randomly sample from a Gaussian (N(0, Variance)) and add the result
        to the dimension dimension.

    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        variance - the variance of a 0 mean Gaussian to add to each point or width of the uniform distribution
        distribution - the distribution to use (gaussian or uniform)
    outputs:
        pc_out - pc with added noise.

    """
    pc_out = []

    if distribution=='gaussian':
        for pt in pc:
            pc_out.append(pt + numpy.random.normal(0, variance, (2, 1)))
    elif distribution=='uniform':
        for pt in pc:
            pc_out.append(pt + numpy.random.uniform(-variance, variance, (2, 1)))
    else:
        raise ValueError(['Unknown distribution type: ', distribution])
    return pc_out


def merge_clouds(pc1, pc2):
    """Add Gaussian noise to pc.

    Merge two point clouds

    inputs:
        pc1 - a list of numpy 2 x 1 matrices that represent one set of points.
        pc2 - a list of numpy 2 x 1 matrices that represent another set of points.
    outputs:
        pc_out - merged point cloud

    """
    pc_out = pc1
    for pt in pc2:
        pc_out.append(pt)

    return pc_out

def add_outliers(pc, multiple_of_data, variance, distribution='gaussian'):
    """Add outliers to pc.

    inputs:
        pc - a list of numpy 2 x 1 matrices that represent the points.
        multiple_of_data - how many outliers to add in terms of multiple of data. Must be an integer >= 1.
        variance - the variance of a 0 mean Gaussian to add to each point.
        distribution - the distribution to use (gaussian or uniform)
    outputs:
        pc_out - pc with added outliers.

    """
    pc_out = pc
    for i in range(0,multiple_of_data):
        pc_outliers = add_noise(pc_out, variance,distribution)
        pc_out = merge_clouds(pc_out,pc_outliers)
    return pc_out

def add_outliers_centroid(pc, num_outliers, variance, distribution='gaussian'):
    """Add outliers to pc (reference to centroid).


    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        num_outliers - how many outliers to add
        variance - the variance of a 0 mean Gaussian to add to each point.
        distribution - the distribution to use (gaussian or uniform)
    outputs:
        pc_out - pc with added outliers.

    """
    centroid = numpy.zeros((2, 1))
    for pt in pc:
        centroid = centroid + pt
    centroid = centroid/len(pc)

    newpoints = []
    for i in range(0,num_outliers):
        newpoints.append(numpy.matrix(centroid))

    return merge_clouds(pc, add_noise(newpoints,variance,distribution))


def convert_pc_to_matrix(pc):
    """Coverts a point cloud to a numpy matrix.

    Inputs:
        pc - a list of 2 by 1 numpy matrices.
    outputs:
        numpy_pc - a 2 by n numpy matrix where each column is a point.

    """
    numpy_pc = numpy.matrix(numpy.zeros((2, len(pc))))

    for index, pt in enumerate(pc):
        numpy_pc[0:2, index] = pt

    return numpy_pc


def convert_matrix_to_pc(numpy_pc):
    """Coverts a numpy matrix to a point cloud (useful for plotting).

    Inputs:
        numpy_pc - a 2 by n numpy matrix where each column is a point.
    outputs:
        pc - a list of 2 by 1 numpy matrices.


    """
    pc = []

    for i in range(0,numpy_pc.shape[1]):
        pc.append((numpy_pc[0:2,i]))

    return pc