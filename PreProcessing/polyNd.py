# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 17:18:46 2018

@author: y0090873
"""
import numpy as np
from scipy.interpolate import lagrange
import itertools


class polyNd(object):

    def __init__(self, polyarray):
        self.dimension=len(polyarray)
        self.polyproduct1d=np.empty(self.dimension, np.poly1d)
        self.order=np.zeros(self.dimension)
        for i in range(self.dimension):
            self.polyproduct1d[i]=polyarray[i]
            self.order[i]=polyarray[i].order

    @staticmethod
    def tensorProductPoly1d(poly1darraylist):
        dimension=len(poly1darraylist)
        numPoly=np.zeros(dimension, 'int32')
        for i in range(dimension):
            numPoly[i]=len(poly1darraylist[i])
        tensorpoly=np.empty(numPoly, polyNd)
        a=np.empty(dimension, np.poly1d)
        for I in itertools.product(*[range(n) for n in numPoly]):
            for j in range(dimension):
                a[j]=(poly1darraylist[j])[I[j]]
            tensorpoly[I]=polyNd(a)
        return tensorpoly

    def value(self, Coordlist):
        value=1.0
        for i in range(self.dimension):
            value=value*(self.polyproduct1d[i](Coordlist[i]))

        return value

    def partialderivative(self, variable=[]):

        b=np.empty(self.dimension, np.poly1d)

        for j in range(0, self.dimension):
            poly1d=self.polyproduct1d[j]
            coeffpoly1d=poly1d.coeffs
            order=poly1d.order
            a=[None]*self.dimension
            for i in range(self.dimension):
                a[i]=self.polyproduct1d[i]
            coeff=[None]*order
            for i in range(order-1,-1,-1):
                coeff[order-i-1]=coeffpoly1d[order-i-1]*(i+1)
            derivpoly1d=np.poly1d(coeff)
            a[j]=derivpoly1d
            b[j]=polyNd(a)

        if variable==[]:
            return b
        else:
            return b[variable-1]

    @staticmethod
    def get_value_for_polyNdarray(polyNdarray, Coordlist):
        l=len(polyNdarray)
        value=np.zeros(l)
        for i in range(l):
            value[i]=polyNdarray[i].value(Coordlist)
        return value

    @staticmethod
    def multiplypolyNd(polyNdarray):
        numPoly=len(polyNdarray)
        dimension=polyNdarray[0].dimension
        a=[None]*dimension
        for i in range(dimension):
            poly=np.poly1d([1])
            for j in range(numPoly):
                poly=poly*(polyNdarray[j].polyproduct1d[i])
            a[i]=poly
        return polyNd(a)
