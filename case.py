# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:46:23 2019

@author: rajuc
"""
import numpy as np
from PreProcessing import Boundaries
from ExecuteExamples import executeCase

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dim=2 # dimension
N=dim*[5, ]
p=dim*[3, ]
origin=np.zeros(dim)
length=0.5*np.ones(dim)
solverType=2
problemType=2

boundary=[]
if dim==2:
    boundary.append(Boundaries.boundary([0., 0.], [0.5, 0.]))
    boundary.append(Boundaries.boundary([0., 0.5], [0.5, 0.5]))
elif dim==3:
    boundary.append(Boundaries.boundary([0., 0., 0.], [0.5, 0., 0.5]))
    boundary.append(Boundaries.boundary([0., 0.5, 0.], [0.5, 0.5, 0.5]))

dirichlet=[]
dirichlet.append({'boundary': boundary[0], 'value': lambda coord: 293})
dirichlet.append({'boundary': boundary[1], 'value': lambda coord: 283})

numQuadPts=dim*[3, ]
solver=executeCase.execute(dim, N, p, origin, length, problemType, solverType, dirichlet, numQuadPts)

# boundary3 = Boundaries.boundary([0., 0.1], [0., 0.4])
# boundary3 = Boundaries.boundary([0.,0.,0.], [0., 1., 1.])
# boundary4 = Boundaries.boundary([1., 0., 0.], [1., 1., 1.])
# periodic_boundary = Boundaries.periodic_boundaries(boundary3, boundary4)
# MyGrid.make_periodic_boundaries(periodic_boundary)

print('END')



### SOLUTION PLOT
Soll=np.reshape(solver.grid.solution_array,[p[0]*N[0]+1,p[1]*N[1]+1])
print (Soll)
x =solver.grid.coord[0,:,1]
y = solver.grid.coord[0,:,1]
X, Y = np.meshgrid(x, y)
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
ha.plot_surface(X, Y, Soll)
#plt.imshow(Soll)
plt.show()