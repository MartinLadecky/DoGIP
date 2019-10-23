# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:30:31 2019

@author: rajuc
"""

import numpy as np

class boundary(object):
    def __init__(self, minCoord, maxCoord):
        assert(len(minCoord) == len(maxCoord))
        self.minCoord = np.asarray(minCoord)
        self.maxCoord = np.asarray(maxCoord)
        self.dimension = len(minCoord) - 1
        check = (self.minCoord == self.maxCoord)
        assert(np.sum(check) == 1)
        self.fixedDim = np.asarray(np.where(check)[0]) + 1

class periodic_boundaries(object):
    def __init__(self, boundary1, boundary2):
        self.boundary1 = boundary1
        self.boundary2 = boundary2
        assert(boundary1.dimension == boundary2.dimension)
        minCheck = (self.boundary1.minCoord != self.boundary2.minCoord)
        maxCheck = (self.boundary1.maxCoord != self.boundary2.maxCoord)
        assert(np.sum(minCheck) == 1)
        assert(np.sum(maxCheck) == 1)
        assert(self.boundary1.fixedDim == self.boundary2.fixedDim)
        self.fixedDim = self.boundary1.fixedDim
