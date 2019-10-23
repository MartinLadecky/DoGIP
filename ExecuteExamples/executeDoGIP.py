# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 06:43:21 2019

@author: rajuc
"""

from Solvers import solverDoGIP


def executeSolver(MyGrid, problemType, dirichlet, numQuadPts):

    MySolver=solverDoGIP.SolverDoGIP(MyGrid, problem_type=problemType)
    MySolver.set_numQuadPts_for_bilinear(numQuadPts)

    MySolver.build_A_DoGIP()
    MySolver.build_interpolation_operators()
    MySolver.build_global_source_vector()

    MySolver.impose_dirichlet_BC(dirichlet)
    MySolver.solve_linear_system()

    return MySolver
