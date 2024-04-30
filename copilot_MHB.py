import numpy as np
import open3d as o3d
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def laplace_beltrami_operator(mesh):
    # Convert the TriangleMesh to a PointCloud
    pcd = mesh.sample_points_poisson_disk(number_of_points=3000)
    
    # Compute the vertex normals
    pcd.estimate_normals()
    
    # Compute the Laplace-Beltrami operator
    L = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)[0].compute_adjacency_list()
    
    return L


def manifold_harmonics_basis(mesh, num_eigenvectors):
    # Compute the Laplace-Beltrami operator
    L = laplace_beltrami_operator(mesh)
    
    # Compute the eigenvectors of the Laplace-Beltrami operator
    _, eigenvectors = eigsh(L, k=num_eigenvectors, which='SM')
    
    return eigenvectors

def visualize_mesh(mesh, scalar_field):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.asarray(mesh.vertices)[:,0], np.asarray(mesh.vertices)[:,1], np.asarray(mesh.vertices)[:,2], c=scalar_field)
    plt.show()

# Load the mesh
mesh = o3d.io.read_triangle_mesh(r'/home/markus/Downloads/LIRIS_EPFL_GenPurpose/Dyno_models/dinosaur-75.obj')

# Compute the manifold harmonics basis
eigenvectors = manifold_harmonics_basis(mesh, num_eigenvectors=10)
print(eigenvectors)

# Visualize the mesh with the first eigenvector as a scalar field
visualize_mesh(mesh, eigenvectors[:,0])

