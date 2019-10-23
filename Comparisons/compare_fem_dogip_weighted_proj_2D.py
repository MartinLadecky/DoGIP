# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:19:11 2019

@author: rajuc
"""

import numpy as np
from get_operators import get_global_stiffness_matrix, get_B_DoGIP, get_B_DoGIP_1D

# N --> Number of Elements in each direction
# Created as a list of Values to study different cases
N_list=[96, 48, 24, 12]

# N --> Order of Polynomials in each direction
# Created as a list of Values to study different cases
p_list=[1, 2, 4, 8]

# Parameters of the problem
dimension=2
origin=[0.0, 0.0]
length=[1.0, 1.0]

threshold=1e-10

# --> Operators
nnz=lambda B, threshold=threshold: np.sum(np.abs(B)>threshold)
mem_sparse=lambda B: 2*nnz(B)+B.shape[0]
ones=lambda B: np.sum(np.abs(np.abs(B.toarray())-1)<threshold)

# --> Comparison of memory requirements and Computational Effectiveness for 2D
#     Weighted Projection between Original and Full Interpolation
print('\tN\t p\t Memory A\t Memory A_DoGIP\t Memmory Eff.\t Comp. eff\t Comp. eff_1D')
#
for i in range(len(N_list)):

    N=N_list[i]
    p=p_list[i]

    # --> Getting the full Interpolation Matrix
    B_hat_full=get_B_DoGIP(2, np.array([p, p]), 1)

    # --> Getting the 1D Interpolation Matrix
    B_hat_1D=get_B_DoGIP_1D(p, 1)

    # Getting the global stiffness matrix
    K=get_global_stiffness_matrix(dimension, origin, length, [N, N], [p, p], 1)

    # --> Computing the number of non -zeros in the B_hat matrix
    nnz_B_hat_full=nnz(B_hat_full)
    nnz_B_hat_1D=nnz(B_hat_1D)
    nnz_K=nnz(K)

    # --> Computing the memory requirements
    mem_K=2*nnz_K+(N*p+1)**2
    mem_A_DoGIP=((2*p+1)**2)*(N**2)

    # --> Computing the computational Complexity
    num_ops_full=(2*nnz_B_hat_full+(p*2+1)**2)*(N**2)
    num_ops_1D=(2*nnz_B_hat_1D*(3*p+2)+(p*2+1)**2)*(N**2)

    # --> Computing Memory efficiency and computational effectiveness
    mem_eff=mem_A_DoGIP/mem_K
    comp_eff=num_ops_full/nnz_K
    comp_eff_1D=num_ops_1D/nnz_K

    print('{:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}'.format(N, p, mem_K, mem_A_DoGIP, mem_eff, comp_eff, comp_eff_1D))
