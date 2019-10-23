# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:42:40 2019

@author: vemburaj
"""

import numpy as np
from Solvers.solverFEM import SolverFEM
from Solvers.solverDoGIP import SolverDoGIP
from Solvers.solverDoGIP_1D import SolverDoGIP_1D_Interpolation
from PreProcessing import preProcessing


# --> Function to get the global stiffness matrix
def get_global_stiffness_matrix(dimension, origin, length, N, p, problemType):

    MyDomain=preProcessing.Domain(dimension)
    MyDomain.set_origin(origin)
    MyDomain.set_domain_length(length)

    MyGrid=preProcessing.Discretization(MyDomain)
    MyGrid.set_num_elements(N)
    MyGrid.set_order(p)
    MyGrid.generate_mesh()
    MyGrid.initialize_arrays(solution_approach=1, problem_type=problemType)

    MySolver=SolverFEM(MyGrid, problem_type=problemType)

    return MySolver.get_dummy_global_stiff_matrix()


def get_B_DoGIP(dimension, p, problemType):

    MyDomain=preProcessing.Domain(dimension)
    MyDomain.set_origin(np.zeros(dimension))
    MyDomain.set_domain_length(np.ones(dimension))

    MyGrid=preProcessing.Discretization(MyDomain)
    MyGrid.set_num_elements(np.ones(dimension, dtype='int32'))
    MyGrid.set_order(p)
    MyGrid.generate_mesh()
    MyGrid.initialize_arrays(solution_approach=2, problem_type=problemType)

    MySolver=SolverDoGIP(MyGrid, problem_type=problemType)
    MySolver.build_interpolation_operators()

    return MySolver.get_interpolation_operators()


def get_B_DoGIP_1D(p, problemType):

    MyDomain=preProcessing.Domain(2)
    MyDomain.set_origin(np.zeros(2))
    MyDomain.set_domain_length(np.ones(2))

    MyGrid=preProcessing.Discretization(MyDomain)
    MyGrid.set_num_elements(np.ones(2, dtype='int32'))
    MyGrid.set_order([p, p])
    MyGrid.generate_mesh()
    MyGrid.initialize_arrays(solution_approach=3, problem_type=problemType)

    MySolver=SolverDoGIP_1D_Interpolation(MyGrid, problem_type=problemType)
    MySolver.build_interpolation_operators()

    return MySolver.get_interpolation_operators()
