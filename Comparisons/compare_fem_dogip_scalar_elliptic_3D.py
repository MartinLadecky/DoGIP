# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:56:50 2019

@author: rajuc
"""

import numpy as np
from get_operators import get_global_stiffness_matrix, get_B_DoGIP, get_B_DoGIP_1D

# N --> Number of Elements in each direction
# Created as a list of Values to study different cases
N_list=[5]
# N --> Order of Polynomials in each direction
# Created as a list of Values to study different cases
p_list=[10]

# Parameters of the problem
dimension=3
origin=[0.0, 0.0, 0.0]
length=[1.0, 1.0, 1.0]

threshold=1e-10

# --> Operators
nnz=lambda B, threshold=threshold: np.sum(np.abs(B)>threshold)
mem_sparse=lambda B: 2*nnz(B)+B.shape[0]
ones=lambda B: np.sum(np.abs(np.abs(B.toarray())-1)<threshold)

# --> Comparison of memory requirements and Computational Effectiveness for 2D
#     Weighted Projection between Original and Full Interpolation
# print('\tN\t p\t Memory A\t Memory A_DoGIP\t Memmory Eff.\t Comp. eff\t Comp. eff_1D')
#
for i in range(len(N_list)):

    N=N_list[i]
    p=p_list[i]

    # --> Getting the full Interpolation Matrix
    B_hat_full=get_B_DoGIP(3, np.array([p, p, p]), 2)

    # --> Getting the 1D Interpolation Matrix
    B_hat_1D_1, B_hat_1D_2=get_B_DoGIP_1D(p, 2)

    # Getting the global stiffness matrix
    K=get_global_stiffness_matrix(dimension, origin, length, [N, N, N], [p, p, p], 2)

    # --> Computing the number of non -zeros in the B_hat matrix
    nnz_B_hat_full=[nnz(B_hat_full[0]), nnz(B_hat_full[1]), nnz(B_hat_full[2])]
    nnz_B_hat_1D_1=int(nnz(B_hat_1D_1))
    nnz_B_hat_1D_2=int(nnz(B_hat_1D_2))

    nnz_K=int(nnz(K))
    # --> Computing the memory requirements
    mem_K=2*nnz_K+(N*p+1)**3
    mem_K_T=((p+1)**3)*((p+1)**3)
    mem_A_DoGIP=9*((2*p+1)**3)*(N**3)
    mem_A_T_DoGIP=9*((2*p+1)**3)

    # --> Computing the computational Complexity
    num_ops_full=(2*(nnz_B_hat_full[0]+nnz_B_hat_full[1]+nnz_B_hat_full[2])+9*(2*p+1)**3)*(N**3)
    # num_ops_full = (2 * nnz_B_hat_full + (p*2 + 1)**2) * (N**2)
    num_ops_1D=(2*(2*nnz_B_hat_1D_1+nnz_B_hat_1D_2)*((p+1)**2+(p+1)*(2*p+1)+(2*p+1)**2)+9*(p*2+1)**3)*(N**3)
    num_ops_1D_alt=(2*3*(nnz_B_hat_1D_1*((p+1)*(2*p+1)+(2*p+1)**2)+nnz_B_hat_1D_2*(p+1)**2)+9*(p*2+1)**3)*(N**3)

    # --> Computing Memory efficiency and computational effectiveness
    mem_eff=(mem_A_DoGIP)/mem_K
    comp_eff=(num_ops_full)/nnz_K
    comp_eff_1D=(num_ops_1D)/nnz_K
    comp_eff_1D_alt=num_ops_1D_alt/nnz_K

    ss=''
    for val in [N, p, mem_K, mem_K_T, mem_A_DoGIP, mem_A_T_DoGIP,
                int(nnz_B_hat_full[0]+nnz_B_hat_full[1]+nnz_B_hat_full[2]),
                mem_eff, comp_eff]:
        if isinstance(val, int):
                ss+=' {:,} &'.format(val)
        elif isinstance(val, float):
                ss+=' %.2f &'%val

    ss=ss[:-1]
    ss+=' \\\\ \n'

    print(ss)

    ss=''
    for val in [N, p, mem_K, mem_K_T, mem_A_DoGIP, mem_A_T_DoGIP,
                nnz_B_hat_1D_1, nnz_B_hat_1D_2, mem_eff, comp_eff_1D]:
        if isinstance(val, int):
                ss+=' {:,} &'.format(val)
        elif isinstance(val, float):
                ss+=' %.2f &'%val

    ss=ss[:-1]
    ss+=' \\\\ \n'

    print(ss)

    # print('{:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}\t {:>12}'.format(N,p,mem_K, mem_A_DoGIP, mem_eff, comp_eff, comp_eff_1D))
