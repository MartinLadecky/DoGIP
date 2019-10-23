# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 06:18:08 2019

@author: rajuc
"""
from PreProcessing import preProcessing
from ExecuteExamples import executeFEM, executeDoGIP, executeDoGIP_1D


def execute(dimension, N, p, origin, length, problemType, solverType, dirichlet, numQuadPts):

    MyDomain=preProcessing.Domain(dimension)
    MyDomain.set_origin(origin)
    MyDomain.set_domain_length(length)

    MyGrid=preProcessing.Discretization(MyDomain)
    MyGrid.set_num_elements(N)
    MyGrid.set_order(p)
    MyGrid.generate_mesh()
    MyGrid.initialize_arrays(solution_approach=solverType, problem_type=problemType)

    if solverType==1:
        solver=executeFEM.executeSolver(MyGrid, problemType, dirichlet, numQuadPts)

    elif solverType==2:
        solver=executeDoGIP.executeSolver(MyGrid, problemType, dirichlet, numQuadPts)

    elif solverType==3:
        solver=executeDoGIP_1D.executeSolver(MyGrid, problemType, dirichlet, numQuadPts)

    return solver
