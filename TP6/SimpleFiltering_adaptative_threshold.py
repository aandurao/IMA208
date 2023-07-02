# -*- coding: utf-8 -*-
"""circum_sphere

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YncydsXf5p5nAbBU8vn3qlouo82Y8f18
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import distance

import math

def distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def circumradius(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    a = distance(x1, y1, z1, x2, y2, z2)
    b = distance(x2, y2, z2, x3, y3, z3)
    c = distance(x3, y3, z3, x1, y1, z1)
    s = (a + b + c)/2
    A = math.sqrt(s*(s-a)*(s-b)*(s-c))
    r = (a*b*c)/(4*A + 1e-10)
    return r

# (3D)
# Open the file and read the vertices as strings
with open("Bimba.xyz", "r") as f:
    vertex_strings = f.readlines()

# Convert the vertex strings to a NumPy array of shape (N, 3)
points3D = np.zeros((len(vertex_strings), 3))
for i, vertex_str in enumerate(vertex_strings):
    vertex_arr = [float(coord) for coord in vertex_str.strip().split()]
    points3D[i] = vertex_arr

tri = Delaunay(points3D)

with open("bimba_adaptative.stl", "w") as o:
    o.write("solid bimba\n")

    R_tab = []
    for tetra in tri.simplices:
        neighbors = tri.vertex_neighbor_vertices
        x,y,z = points3D[tetra[0]], points3D[tetra[1]], points3D[tetra[2]]
        vertex_density = []
        for vertex in tetra:
            neigh_count = neighbors[0][vertex + 1] - neighbors[0][vertex]
            vertex_density.append(neigh_count)
        density = np.mean(vertex_density)
        gamma = 4.3
        threshold = np.exp(-(density/gamma))
        R = circumradius(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
        if R < threshold:
            o.write("facet normal 0 0 0\n")
            o.write("\touter loop\n")
            o.write(f"\t\tvertex {x[0]} {x[1]} {x[2]}\n")
            o.write(f"\t\tvertex {y[0]} {y[1]} {y[2]}\n")
            o.write(f"\t\tvertex {z[0]} {z[1]} {z[2]}\n")
            o.write("\tendloop\n")
            o.write("endfacet\n")

    # Write the end of the STL file
    o.write("endsolid bimba")

