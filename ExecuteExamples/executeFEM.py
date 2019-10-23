# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 06:37:35 2019

@author: rajuc
"""

from Solvers import solverFEM


def executeSolver(MyGrid, problemType, dirichlet, numQuadPts):

    MySolver=solverFEM.SolverFEM(MyGrid, problem_type=problemType)
    MySolver.set_numQuadPts_for_bilinear(numQuadPts)

    MySolver.build_global_stiffness_matrix()
    MySolver.build_global_source_vector()

    MySolver.impose_dirichlet_BC(dirichlet)
    MySolver.solve_linear_system()

    return MySolver
