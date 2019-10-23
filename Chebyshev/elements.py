import numpy as np
import copy
#import scipy as sc
from ffthompy.tensors.objects import Tensor

from Chebyshev.discrete_cheby import cheb_extrema_grid1D
from Chebyshev.DCTs import * # dctn_, idctn_, grad_,grad_adjoin_, decrease_spectrum, enlarge_spectrum, divide_both ,\
       # dctn_ortho_, idctn_ortho_ ,dctn_trans_, idctn_trans_
from Chebyshev.plot_functions import *
from scipy.interpolate import lagrange
from scipy.integrate import nquad

import itertools

#from Chebyshev.problem import load,mat_fun

class Chebyshev_element(Tensor):

    def __repr__(self, full = False, detailed = False):
        keys = ['order', 'name', 'Y', 'shape', 'N', 'Fourier', 'fft_form', 'origin', 'norm']
        ss = self._repr(keys)
        skip = 4*' '
        if np.prod(np.array(self.shape)) <= 36 or detailed:
            ss += '{0}norm component-wise =\n{1}\n'.format(skip, str(self.norm()))
            ss += '{0}mean = \n{1}\n'.format(skip, str(self.mean()))
        if full:
            ss += '{0}val = \n{1}'.format(skip, str(self.val))
        return ss

    def _set_fft(self, fft_form):
        assert (fft_form in ['c', 'r', 0, 'cheb'])

        self.fft_form = 'cheb'

    def dct(self):
        if not self.Fourier:
            if np.size(self.shape) == 0:
               self.val = dctn_(self.val)
            elif np.size(self.shape) > 0:
                for d in np.arange(np.size(self.shape)+1):
                    self.val[d] = dctn_(self.val[d])
            #self.val = dctn_(self.val)
            self.Fourier = True

            return copy.deepcopy(self)

        else:
            raise ValueError('Tensor is already in spectral space.')

    def dct_ortho(self):
        if not self.Fourier:
            if np.size(self.shape) == 0:
               self.val = dctn_ortho_(self.val)
            elif np.size(self.shape) > 0:
                for d in np.arange(np.size(self.shape)+1):
                    self.val[d] = dctn_ortho_(self.val[d])
            #self.val = dctn_(self.val)
            self.Fourier = True

            return copy.deepcopy(self)

        else:
            raise ValueError('Tensor is already in spectral space.')


    def dct_trans(self):
        if self.Fourier:

            if np.size(self.shape) == 0:
                self.val = dctn_trans_(self.val)
            elif np.size(self.shape) > 0:
                for d in np.arange(np.size(self.shape) + 1):
                    self.val[d] = dctn_trans_(self.val[d])
            # self.val = dctn_(self.val)
            self.Fourier = False

            return copy.deepcopy(self)

        else:
            raise ValueError('Tensor is NOT in spectral space.')

    def idct(self):
        if self.Fourier:
           # for index in np.ndindex(self.shape):
               # self.val[index] = idctn_(self.val[index])
            if np.size(self.shape) == 0:
               self.val = idctn_(self.val)
            elif np.size(self.shape) > 0:
                for d in np.arange(np.size(self.shape)+1):
                    self.val[d] = idctn_(self.val[d])

            self.Fourier = False

            return copy.deepcopy(self)
        else:
            raise ValueError('Tensor is already in physical space.')

    def idct_ortho(self):
        if self.Fourier:
           # for index in np.ndindex(self.shape):
               # self.val[index] = idctn_(self.val[index])
            if np.size(self.shape) == 0:
               self.val = idctn_ortho_(self.val)
            elif np.size(self.shape) > 0:
                for d in np.arange(np.size(self.shape)+1):
                    self.val[d] = idctn_ortho_(self.val[d])

            self.Fourier = False

            return copy.deepcopy(self)
        else:
            raise ValueError('Tensor is already in physical space.')


    def idct_trans(self):
        if not self.Fourier:
           # for index in np.ndindex(self.shape):
               # self.val[index] = idctn_(self.val[index])
            if np.size(self.shape) == 0:
               self.val = idctn_trans_(self.val)
            elif np.size(self.shape) > 0:
                for d in np.arange(np.size(self.shape)+1):
                    self.val[d] = idctn_trans_(self.val[d])

            self.Fourier = True

            return copy.deepcopy(self)
        else:
            raise ValueError('Tensor is NOT in physical space.')

    def interpolate(self):
        # Interpolate self.val into double grid points
        # Done be dct, extending spectrum, divide middle frequency to keep result correct and idct
        if self.Fourier:
            raise ('Tensor is NOT in Physical space')

        self.dct()
        self.enlarge()
        self.idct()

        return  # return tensor interpolated on double grid

    def enlarge(self):  # enlarge spectrum of tensor by zeros to 2*N-1

        # enlarge spectrum
        if not self.Fourier:
            raise ('Tensor is not in Fourier space')

        dN = tuple(np.array(np.multiply(self.N, 2) - 1, dtype = np.int))
        val = np.zeros(tuple(self.shape) + dN, dtype = self.val.dtype)
       # np.ndindex(self.shape)
        if self.shape.__len__() == 0:
            val = enlarge_spectrum(self.val)

        elif self.shape == (self.dim,):
            for d in range(self.val.shape[0]):
                val[d] = enlarge_spectrum(self.val[d])

        else:
            raise ('Enlarge works only for 0 and 1 order tensors')

        self.val = val
        self.N = dN

        return  # return tensor.val extended to size of double grid

    def shrink(self):  # shrink spectrum to size N+1 from 2N+1
        if not self.Fourier:
            raise ('Tensor is in Physical space')

        #self.dct()

        smaller= Tensor(name = 'gradient', N = np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int), shape = self.shape, multype = 'grad')
        for d in np.ndindex(self.shape):#range(self.dim):
            smaller.val[d] =decrease_spectrum(self.val[d], tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int)))
           # smaller.val[d] =

 #       weights = lagrange_weights(self.N)
 #       for index in np.ndindex(self.shape):
  #          self.val[index] = np.einsum('...,...->...', self.val[index], weights)

        self.val=copy.deepcopy(smaller.val)

        self.N = tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int))
        #self.idct()

        return

    def grad(self):

        if not self.Fourier:
            print('Tensor {} is not in Fourier space'.format(self.name))
            print('Tensor {} is transformed due to gradient'.format(self.name))
            self.dct()

        grad = Tensor(name = 'gradient', N = self.N, shape = [self.dim], multype = 'grad')

        for d in range(self.dim):
            grad.val[d] = grad_(copy.deepcopy(self.val), d)

        self.val = copy.deepcopy(grad.val)
        self.shape = (self.dim,)
        return

    def grad_ortho(self):

        if not self.Fourier:
            print('Tensor {} is not in Fourier space'.format(self.name))
            print('Tensor {} is transformed due to gradient'.format(self.name))
            self.dct_ortho()

        grad = Tensor(name = 'gradient', N = self.N, shape = [self.dim], multype = 'grad')

        for d in range(self.dim):
            grad.val[d] = grad_ortho_(copy.deepcopy(self.val), d)

        self.val = copy.deepcopy(grad.val)
        self.shape = (self.dim,)
        return

    def div_and_decrease(self):

        if not self.Fourier:
            print('Tensor {} is not in Fourier space'.format(self.name))
            print('Tensor {} is transformed due to gradient'.format(self.name))
            self.dct()

        diver = np.zeros(tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int)), dtype = np.float_)

        for d in range(self.dim):
            Ad = decrease_spectrum(self.val[d], tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int)))

          #  Aa = decrease(self.val(d), tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int)))
            Aa = grad_(copy.deepcopy(Ad), d)

            diver = diver + Aa

        diver = idctn_(diver)
        #   Bb = grad_(copy.deepcopy(self.val[d]), d)
         #   Aa = decrease(Bb, tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int)))
         #   Aa = idctn_(Aa)
         #   diver = diver + Aa

        self.val = copy.deepcopy(diver)
        self.N = tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int))
        self.Fourier = False
        self.shape = ()
        return

    def div(self):

        if not self.Fourier:
            print('Tensor {} is not in Fourier space'.format(self.name))
            print('Tensor {} is transformed due to gradient'.format(self.name))
            self.dct()

        diver = np.zeros(tuple(np.array(self.N, dtype = np.int)), dtype = np.float_)
        for d in range(self.dim):
            Bb = grad_(copy.deepcopy(self.val[d]), d)
           # Aa = decrease(Bb, tuple(np.array(np.divide(np.add(self.N, 1), 2), dtype = np.int)))
            Aa = idctn_(Bb)
            diver = diver + Aa

        self.val = copy.deepcopy(diver)
        self.N = tuple(np.array(self.N, dtype = np.int))
        self.Fourier = False
        self.shape = ()
        return

    def grad_transpose(self):

        if not self.Fourier:
            print('Tensor {} is not in Fourier space'.format(self.name))
            print('Tensor {} is transformed due to gradient'.format(self.name))
            self.dct()

        adjoin = np.zeros(tuple(np.array(self.N, dtype = np.int)), dtype = np.float_)
   #     for index in np.ndindex(self.shape):
   #         print(index)

        for d in range(self.dim):#np.ndindex(self.shape):#:
            Bb = grad_adjoin_(copy.deepcopy(self.val[d]), d)
         #   Aa = idctn_(Bb)
           # print(Bb)
            adjoin = adjoin +Bb# Aa#Aa

        self.val = copy.deepcopy(adjoin)
        self.N = tuple(np.array(self.N, dtype = np.int))
        #self.Fourier = True
        self.shape = ()

        return






    def integrate(self):
        weights = clenshaw_curtis_weights(self.N)
        # integral=copy.deepcopy(self)
        integral = np.einsum('...,...->...', self.val, weights)
        #      if integral.dim == 2:
        #          for i, j in np.ndindex(integral.shape):
        #              integral.val[i][j]=np.einsum('...,...->...', integral.val[i][j], weights)
        #      elif integral.dim == 3:
        #          for i, j, k in np.ndindex(integral.shape):
        #              integral.val[i][j][k]=np.einsum('...,...->...', integral.val[i][j][k], weights)

        self.integral = integral.sum()


    def Gauss_integrateS(self,material_fun):
        #for index in np.ndindex(self.shape):

        for ki, kj in itertools.product(np.arange(0, self.N[0]), np.arange(0, self.N[1])):
            k = [ki, kj]
            print(k)

            l_k = lambda x0, x1: material_fun*lagrange_d([x0, x1], self.N, k)

            int = nquad(l_k, [[-1, 1], [-1, 1]])
            print(int)
            self.val[ki, kj] = int[0]
            print(self.val)

    def Gauss_integrateA(self, material_fun):
        # for index in np.ndindex(self.shape):
        for index in [0, 0], [1, 1]:
            for ki, kj in itertools.product(np.arange(0, self.N[0]), np.arange(0, self.N[1])):
                k = [ki, kj]
                print(k)

                l_k = lambda x0, x1: material_fun*lagrange_d([x0, x1], self.N, k)

                int = nquad(l_k, [[-1, 1], [-1, 1]])
                print(int)
                self.val[index[0], index[1], ki, kj] = int[0]
                print(self.val)


    def set_nodal_coord(self):
        self.coord = np.meshgrid(*[cheb_extrema_grid1D(self.N[d]) for d in range(0, self.dim)], indexing = 'ij')

    def set_dg_coord(self):  # s
        self.coord = np.meshgrid(*[cheb_extrema_grid1D(self.N[d] + self.N[d] - 1) for d in range(0, self.dim)],
                                 indexing = 'ij')

    def set_val(self, values):
        if isinstance(values, np.ndarray):  # define: val + orde..r
            self.val = values

    def plot_val(self):
        plot_field(self.coord, self.val, self.dim)

    def plot_grad(self):
        for d in range(self.dim):
            plot_field(self.coord, self.val[d], self.dim)

    def mul_by_cc_weights(self):
        # if not self.shape.__len__() == 2:
        #      raise('work onli')

        weights = clenshaw_curtis_weights(self.N)

        for index in np.ndindex(self.shape):
            self.val[index] = np.einsum('...,...->...', self.val[index], weights)

    def mul_by_lagrange_weights(self):

        weights = lagrange_weights(self.N)
        for index in np.ndindex(self.shape):
            self.val[index] = np.einsum('...,...->...', self.val[index], weights)

def clenshaw_curtis_weights(N):
    a_i = np.zeros(N[0], dtype = np.double)
    for j in range(0, N[0], 2):
        a_i[j] = 2/(1 - j**2)

    w = Chebyshev_element(name = 'weights', N = N, shape = (), multype = 'scal')
    if N.__len__() == 2:
        w.val = np.einsum('i,j->ij', a_i, a_i)
    elif N.__len__() == 3:
        w.val = np.einsum('i,j,k->ijk', a_i, a_i, a_i)

    w.dct()
    w.val = divide_both(w.val)
    w.Fourier = False

    return w.val

def lagrange_weights(N):

    x = np.array(cheb_extrema_grid1D(N[0]))
    y = np.array(cheb_extrema_grid1D(N[1]))

    field = np.arange(-1, 1.1, 0.1)
    integrale = np.zeros(N)

    for xi, yi in itertools.product(np.arange(0,N[0]), np.arange(0,N[1])):
        f_x = np.zeros(np.size(x))
        f_x[xi] = 1
        lag_polx = lagrange(x, f_x)

        f_y = np.zeros(np.size(y))
        f_y[yi] = 1
        lag_poly = lagrange(y, f_y)

        discrete_x = lag_polx(field)
        discrete_y = lag_poly(field)

        discrete = np.outer(discrete_x, discrete_y)


        int = discrete.sum()/(np.size(discrete))
        integrale[xi,yi]=int

    return integrale



def lagrange_d(x,N,k):

    x_dp = np.array(cheb_extrema_grid1D(N[0]))
    y_dp = np.array(cheb_extrema_grid1D(N[1]))

    f_x = np.zeros(np.size(x_dp))
    f_x[k[0]] = 1
    lag_polx = lagrange(x_dp, f_x)

    f_y = np.zeros(np.size(y_dp))
    f_y[k[1]] = 1
    lag_poly = lagrange(y_dp, f_y)

    ff_x = lag_polx(x[0])
    ff_y = lag_poly(x[1])
    #print(ff_x*ff_y)
    return ff_x*ff_y





if __name__ == '__main__':
    print('end')
    dim=2
    n=[2]
    N = np.array(dim*[5, ], dtype = np.int)

  #  RHS= Chebyshev_element(name = 'RHS', N = N, shape = (), multype = 'scal')
  #  RHS.set_nodal_coord()
  #  sol_val = lambda x: 10+x[0]*0#2*x[0]**2 + 2*x[1]**2 - 4
  #  RHS.val = sol_val(RHS.coord)

#   x_points=np.meshgrid(*[np.arange(-1, 1.05, 0.05) for d in range(0, dim)], indexing = 'ij')
#    k=[2,3]
#    f_x = lagrange_d(x_points, N, k)
#   plot_field(x_points, f_x, dim)
#    plt.show()

    Integr = Chebyshev_element(name = ' Integr ', N = N, shape = (), multype = 'scal')
    Integr.set_nodal_coord()
    for ki, kj in itertools.product(np.arange(0, N[0]), np.arange(0, N[1])):
        k = [ki, kj]
        print(k)
       # x_points = np.meshgrid(*[np.arange(-1, 1.05, 0.05) for d in range(0, dim)], indexing = 'ij')
       # f_x = lagrange_d(x_points, N, k)
       # plot_field(x_points, f_x, dim)
       # plt.show()

        l_k = lambda x0,x1: 10*lagrange_d([x0,x1], N, k)

        int=nquad(l_k, [[-1,1], [-1,1]])
        print(int)
        Integr.val[ki,kj]=int[0]
        print(Integr.val)

    plot_field(Integr.coord, Integr.val, dim)
    plt.show()
    a=5