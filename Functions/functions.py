# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:26:15 2019

@author: rajuc
"""

import numpy as np


def material_matrix_value(Coord):
    dim=len(Coord)
    value=np.zeros((dim, dim))
    for i in range(dim):
        value[i, i]=1e11
    return np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])

def material_scalar_value(Coord):
    value=2
    return value

def source_function_value(Coord):
    value=0
    return value
