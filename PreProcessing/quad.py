# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:43:41 2018

@author: y0090873
"""

import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.legendre import leggauss
import itertools
import copy


class Quadrature(object):

    def __init__(self, numpoints):
        self.numpoints=numpoints

    @staticmethod
    def get_1d_gauss_quad(numpoints):
        return leggauss(numpoints)
#        poly = np.poly1d([1,0,-1])
#        b = np.poly1d([1])
#        for i in range(numpoints):
#            b = b * poly
#        coeff = b.coeffs
#        length = len(coeff)
#        for i in range(numpoints):
#            k = length-i-1
#            for j in range(length-i-1):
#                coeff[j] = coeff[j] * k
#                k = k - 1
#        derivpoly = np.poly1d(coeff[0:numpoints+1])
#        quadpoints = derivpoly.r
#        V = np.ones((numpoints, numpoints))
#        g = np.zeros(numpoints)
#        g[0] = 2.
#        for i in range(1, numpoints, 1):
#            V[i,:] = np.power(quadpoints, i)
#            if i % 2 == 0:
#                g[i] = 2. / (i+1)
#        weights = np.linalg.solve(V,g)
#

    @staticmethod
    def get_gauss_quad(numpointslist):

        if not isinstance(numpointslist, np.ndarray):
            numpointslist=np.asarray(numpointslist)

        dimension=max(numpointslist.shape)
        dim=np.hstack((numpointslist, dimension))
        gausspoints=np.zeros(dim)
        weights=np.ones(dim[:-1])

        index=[None]*(dimension+1)
        tile_size=copy.deepcopy(dim)
        for i in range(dimension):
            [gausspoints_1d, weights_1d]=Quadrature.get_1d_gauss_quad(numpointslist[i])
            vec_size=np.ones(dimension+1, dtype='int32')
            vec_size[i]=numpointslist[i]
            gausspoints_1d=np.reshape(gausspoints_1d, vec_size)
            weights_1d=np.reshape(weights_1d, vec_size[:-1])
            tile_size[i]=1
            tile_size[-1]=1
            for j in range(dimension):
                index[j]=slice(0, numpointslist[j], 1)
            index[-1]=slice(i, i+1, 1)
            gausspoints[tuple(index)]=np.tile(gausspoints_1d, tile_size)
            weights[tuple(index[:-1])]=np.multiply(weights, np.tile(weights_1d, tile_size[:-1]))
            tile_size[i]=dim[i]

        return gausspoints, weights
