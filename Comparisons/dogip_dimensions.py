# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:33:44 2019

@author: vemburaj
"""

import numpy as np
from FEM import RectReferenceElement as ReferenceElement
from FEM import DoGIP as DoGIP
import itertools

# N --> Number of Elements in each direction
# Created as a list of Values to study different cases
N_list=[10, 20, 30, 50, 100]

# N --> Order of Polynomials in each direction
# Created as a list of Values to study different cases
p_list=[1, 2, 4, 6, 8, 10]

threshold=1e-10

# --> Operators
nnz=lambda B, threshold=threshold: np.sum(np.abs(B)>threshold)
mem_sparse=lambda B: 2*nnz(B)+B.shape[0]
ones=lambda B: np.sum(np.abs(np.abs(B.toarray())-1)<threshold)

# --> Comparison of memory requirements and Computational Effectiveness for 2D
#     Weighted Projection between Original and Full Interpolation
print('N\t p\t Complexity (Full)\t Complexity (ID)\t Effectiveness')

for N, p in itertools.product(N_list, p_list):

  # --> Getting the full Interpolation Matrix
  B_hat_full=get_B_hat(2, np.array([p, p]))

  # --> Getting the 1D Interpolation Matrix
  B_hat_1D=get_B_hat(1, np.array([p]))

  # --> Computing the number of non -zeros in the B_hat matrix
  nnz_B_hat_full=nnz(B_hat_full)-ones(B_hat_full)
  nnz_B_hat_1D=nnz(B_hat_1D)-ones(B_hat_1D)

  # --> Computing the computational Complexity
  num_ops_full=(2*nnz_B_hat_full+(p*2+1)**2)*(N**2)
  num_ops_1D=(2*nnz_B_hat_1D*(3*p+2)+(p*2+1)**2)*(N**2)
  eff=num_ops_1D/num_ops_full

  print('{:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}'.format(N, p, num_ops_full, num_ops_1D, eff))

print('\n')
print('N\t p\t Complexity (Full)\t Complexity (ID)\t Effectiveness')

for N, p in itertools.product(N_list, p_list):

  # --> Getting the full Interpolation Matrix
  B_hat_full=get_B_hat(3, np.array([p, p, p]))

  # --> Getting the 1D Interpolation Matrix
  B_hat_1D=get_B_hat(1, np.array([p]))

  # --> Computing the number of non -zeros in the B_hat matrix
  nnz_B_hat_full=nnz(B_hat_full)-ones(B_hat_full)
  nnz_B_hat_1D=nnz(B_hat_1D)-ones(B_hat_1D)

  # --> Computing the computational Complexity
  num_ops_full=(2*nnz_B_hat_full+(p*2+1)**3)*(N**3)
  num_ops_1D=(2*nnz_B_hat_1D*((p+1)**2+(2*p+1)**2+(p+1)*(2*p+1))+(p*2+1)**3)*(N**3)
  eff=num_ops_1D/num_ops_full

  print('{:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}'.format(N, p, num_ops_full, num_ops_1D, eff))

