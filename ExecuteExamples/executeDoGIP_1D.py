# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 06:44:50 2019

@author: rajuc
"""

from Solvers import solverDoGIP_1D


def executeSolver(MyGrid, problemType, dirichlet, numQuadPts):

    MySolver=solverDoGIP_1D.SolverDoGIP_1D_Interpolation(MyGrid, problem_type=problemType)
    MySolver.set_numQuadPts_for_bilinear(numQuadPts)

    MySolver.build_A_DoGIP()
    MySolver.build_interpolation_operators()
    MySolver.build_global_source_vector()

    MySolver.impose_dirichlet_BC(dirichlet)
    MySolver.solve_linear_system()

    return MySolver
