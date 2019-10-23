from Chebyshev.elements import Chebyshev_element, clenshaw_curtis_weights
import numpy as np
import copy
from ffthompy.tensors.objects import Tensor
from ffthompy.general.solver import linear_solver, minimal_residual
from ffthompy import Struct

import matplotlib.pyplot as plt
from Chebyshev.discrete_cheby import cheb_extrema_grid1D
from Chebyshev.DCTs import dctn_, idctn_, grad_, decrease_spectrum, enlarge_spectrum, divide_both
from Chebyshev.plot_functions import plot_field


def proble_Ga_Ni(N, mat, f, useCC = True):
    dim = N.size
    dN = tuple(np.array(N*2 - 1, dtype = np.int))

    x0 = Chebyshev_element(name = 'x', N = N, shape = (), multype = 'scal')
    x0.set_nodal_coord()
    #  fun_val = lambda x: ((x[0]**4)*x[1]**3) - x[1]**3 - ((x[0]**4)*x[1]) + x[1] if x.__len__() == 2 else (x[0]**4) + (x[1]**3) + np.cos(x[2])
    #  x0.set_val(fun_val(x0.coord))

    material = Chebyshev_element(name = 'A_mat', N = N, shape = 2*[dim], multype = '21')
    material.set_nodal_coord()
    material.val = mat_fun(material.coord, kind = mat)
    material.mul_by_cc_weights()

    RHS = Chebyshev_element(name = 'RHS', N = N, shape = (), multype = 'scal')
    RHS.set_nodal_coord()
    # sol.val = sol_val(sol.coord)
    RHS.set_val(load(RHS.coord, kind = f))
    RHS.mul_by_cc_weights()
    #   print('integral of RHS full {}'.format(RHS.val.sum()))
    DBC_0(RHS)

    def DivAgrad(x, A):
        x = copy.deepcopy(x)
        #        print('x={}'.format(x.val))
        x.dct()
        #        print('Fx={}'.format(x.val))
        x.grad()
        #        print('GFx={}'.format(x.val))
        # x.enlarge()
        x.idct()
        #        print('FGFx={}'.format(x.val))
        x.val = (A*x).val
        #        print('AFGFx={}'.format(x.val))
        x.idct_trans()
        #        print('FAFGFx={}'.format(x.val))
        x.grad_transpose()
        #        print('GFAFGFx={}'.format(x.val))
        x.dct_trans()
        #        print('FGFAFGFx={}'.format(x.val))

        # x.mul_by_cc_weights()
        return x

    def mV(x0):
        x = DivAgrad(x0, material)
        DBC_0(x)
        return x

    pars = Struct(dim = dim,  # number of dimensions (works for 2D and 3D)
                  N = dim*(N,),  # number of voxels (assumed equal for all directions)
                  Y = np.ones(dim),  # size of periodic cell
                  solver = dict(tol = 1e-8,
                                maxiter = 10000,
                                alpha = 100,
                                approx_omega = False,
                                divcrit = False,
                                ),
                  )
    # pars.update(Struct(alpha = 0.5*(material.val[0, 0].min() + material.val[0, 0].max())))

    sol, info = linear_solver(solver = 'CG', Afun = mV, B = RHS, x0 = x0, par = pars.solver, callback = None)

    # sol, info   =   minimal_residual(Afun = mV, B = RHS, x0 = x0,par = pars.solver,)
    return sol, info


def proble_Ga(N, mat, f, useCC = True):
    dim = N.size
    dN = tuple(np.array(N*2 - 1, dtype = np.int))

    x0 = Chebyshev_element(name = 'x', N = N, shape = (), multype = 'scal')
    x0.set_nodal_coord()
    #   fun_val = lambda x: ((x[0]**4)*x[1]**3) - x[1]**3 - ((x[0]**4)*x[1]) + x[1] if x.__len__() == 2 else (x[0]**4) + (
    #          x[1]**3) + np.cos(x[2])
    #   x0.set_val(fun_val(x0.coord))

    material = Chebyshev_element(name = 'A_mat', N = dN, shape = 2*[dim], multype = '21')
    material.set_nodal_coord()
    material.val = mat_fun(material.coord, kind = mat)
    material.mul_by_cc_weights()

    RHS = Chebyshev_element(name = 'RHS', N = N, shape = (), multype = 'scal')
    RHS.set_nodal_coord()
    # sol.val = sol_val(sol.coord)
    RHS.set_val(load(RHS.coord, kind = f))
    RHS.mul_by_cc_weights()
    #   print('integral of RHS full {}'.format(RHS.val.sum()))
    DBC_0(RHS)

    def DivAgrad(x, A):
        x = copy.deepcopy(x)
        #        print('x={}'.format(x.val))
        x.dct()
        #        print('Fx={}'.format(x.val))
        x.grad()
        #        print('GFx={}'.format(x.val))
        x.enlarge()
        #        print('PGFx={}'.format(x.val))
        x.idct()
        #        print('FPGFx={}'.format(x.val))
        x.val = (A*x).val
        #      x.mul_by_cc_weights()
        #        print('AFPGFx={}'.format(x.val))
        x.idct_trans()
        #        print('FAFPGFx={}'.format(x.val))
        x.shrink()
        #        print('PFAFPGFx={}'.format(x.val))
        x.grad_transpose()
        #        print('GPFAFPGFx={}'.format(x.val))
        x.dct_trans()
        #        print('FGPFAFPGFx={}'.format(x.val))
        # x.idct()
        #  x.div_and_decrease()
        return x

    def mV(x0):
        x = DivAgrad(x0, material)
        DBC_0(x)
        return x

    pars = Struct(dim = dim,  # number of dimensions (works for 2D and 3D)
                  N = dim*(N,),  # number of voxels (assumed equal for all directions)
                  Y = np.ones(dim),  # size of periodic cell
                  solver = dict(tol = 1e-8,
                                maxiter = 1000,
                                alpha = 100,
                                approx_omega = False,
                                divcrit = False,
                                ),
                  )
    # pars.update(Struct(alpha = 0.5*(material.val[0, 0].min() + material.val[0, 0].max())))

    sol, info = linear_solver(solver = 'CG', Afun = mV, B = RHS, x0 = x0, par = pars.solver, callback = None)

    #  sol, info   =   minimal_residual(Afun = mV, B = RHS, x0 = x0,par = pars.solver,)
    return sol, info


def proble_Colocation(N, mat, f, useCC = True):
    dim = N.size
    dN = tuple(np.array(N*2 - 1, dtype = np.int))

    x0 = Chebyshev_element(name = 'x', N = N, shape = (), multype = 'scal')
    x0.set_nodal_coord()

    material = Chebyshev_element(name = 'A_mat', N = N, shape = 2*[dim], multype = '21')
    material.set_nodal_coord()
    material.val = mat_fun(material.coord, kind = mat)

    RHS = Chebyshev_element(name = 'RHS', N = N, shape = (), multype = 'scal')
    RHS.set_nodal_coord()
    RHS.set_val(-load(RHS.coord, kind = f))
    RHS.mul_by_cc_weights()

    #    print('integral of RHS {}'.format(RHS.val.sum()))
    DBC_0(RHS)

    def DivAgrad_Coll(x, A):
        x = copy.deepcopy(x)
        x.dct()
        x.grad()
        x.idct()
        x.val = (A*x).val
        x.dct()
        x.div()
        x.mul_by_cc_weights()
        return x

    def mV(x0):
        x = DivAgrad_Coll(x0, material)
        DBC_0(x)
        return x

    pars = Struct(dim = dim,  # number of dimensions (works for 2D and 3D)
                  N = dim*(N,),  # number of voxels (assumed equal for all directions)
                  Y = np.ones(dim),  # size of periodic cell
                  recover_sparse = 1,  # recalculate full material coefficients from sparse one
                  solver = dict(tol = 1e-8,
                                maxiter = 1000,
                                alpha = 100,
                                approx_omega = False,
                                divcrit = False,
                                ),
                  )

    # sol, info =  minimal_residual(Afun = mV, B = RHS, x0 = x0,par = pars.solver,)

    sol, info = linear_solver(solver = 'CG', Afun = mV, B = RHS, x0 = x0, par = pars.solver, callback = None)

    return sol, info


def problem_WP0(N, mat, f):
    dim = N.size
    dN = tuple(np.array(N*2 - 1, dtype = np.int))

    x0 = Chebyshev_element(name = 'x', N = N, shape = (), multype = 'scal')
    x0.set_nodal_coord()
    #  fun_val = lambda x: ((x[0]**4)*x[1]**3) - x[1]**3 - ((x[0]**4)*x[1]) + x[1] if x.__len__() == 2 else (x[0]**4) + (
    #           x[1]**3) + np.cos(x[2])
    #   x0.set_val(fun_val(x0.coord))

    # solfun= lambda x:((x[0]**4) * x[1]**3)- x[1]**3-( (x[0]**4) * x[1])+x[1]
    # x0.set_val(solfun(x0.coord))
    #  x0.plot_val()
    #  plt.show()

    material = Chebyshev_element(name = 'A_mat', N = dN, shape = (), multype = 'scal')
    material.set_nodal_coord()
    mat_fun0 = lambda x: (x[0]**2 + x[1]*0)
    material.val = mat_fun0(material.coord)

    #    plot_field(material.coord, material.val, dim)
    #  plot_field(material.coord, material.val[1][1], dim)
    # material.Gauss_integrateA(1)
    #    material.mul_by_cc_weights()#mul_by_cc_weights()

    #    material.mul_by_lagrange_weights()
    #   plot_field(material.coord, material.val[0][0], dim)
    #   plot_field(material.coord, material.val[1][1], dim)
    #   plt.show()

    RHS = Chebyshev_element(name = 'RHS', N = N, shape = (), multype = 'scal')
    RHS.set_nodal_coord()
    # sol.val = sol_val(sol.coord)
    # RHS.set_val(load(RHS.coord,kind=f))
    # test_load = lambda x: 1+x[0]*0
    #    RHS.Gauss_integrateS(1)
    # sol_val = lambda x: 2*x[0]**2+2*x[1]**2-4
    sol_val = lambda x: ((6*(x[0]**6)*x[1]**3) - (3*(x[0]**2)*x[1]) - (6*(x[0]**4)*x[1]) + 6*x[1])
    # ((12*(x[0]**2)*x[1]**3) - (12*(x[0]**2)*x[1]) + (6*(x[0]**4)*x[1]) - 6*x[1])

    RHS.set_val(sol_val(RHS.coord))
    RHS.plot_val()
    plt.show()

    RHS.mul_by_cc_weights()
    #    aa=copy.deepcopy(RHS)
    #    bb = copy.deepcopy(RHS)
    #    aa.mul_by_cc_weights()
    #    bb.mul_by_lagrange_weights()
    #    aa.plot_val()
    #    bb.plot_val()

    print('integralis of RHS full {}'.format(RHS.val.sum()))
    RHS.plot_val()
    plt.show()
    DBC_0(RHS)

    def DivAgrad(x, A):
        x = copy.deepcopy(x)
        x.dct()
        # x.grad()
        x.enlarge()
        x.idct()
        x.val = (A*x).val
        x.dct()
        x.shrink()
        # x.
        return x

    def mV(x0):
        x = DivAgrad(x0, material)
        DBC_0(x)
        return x

    pars = Struct(dim = dim,  # number of dimensions (works for 2D and 3D)
                  N = dim*(N,),  # number of voxels (assumed equal for all directions)
                  Y = np.ones(dim),  # size of periodic cell
                  solver = dict(tol = 1e-8,
                                maxiter = 1000,
                                alpha = 100,
                                approx_omega = False,
                                divcrit = False,
                                ),
                  )
    # pars.update(Struct(alpha = 0.5*(material.val[0, 0].min() + material.val[0, 0].max())))

    sol, info = linear_solver(solver = 'CG', Afun = mV, B = RHS, x0 = x0, par = pars.solver, callback = None)

    # sol, info   =   minimal_residual(Afun = mV, B = RHS, x0 = x0,par = pars.solver,)
    return sol, info


def DBC_0(x, b_kind = 'homo'):  # impose 0 boundary condition
    if b_kind == 'homo':
        for d in range(x.dim):
            index = x.dim*[slice(None, None)]
            index[d] = slice(None, None, x.N[d] - 1)
            x.val[tuple(index)] = 0

    if b_kind == 'nonhomo':
        for d in range(1):
            index = x.dim*[slice(None, None)]
            index[d] = slice(None, None, x.N[d])
            x.val[tuple(index)] = 0

            index = x.dim*[slice(None, None)]
            index[d] = slice(x.N[d] - 1, x.N[d] + 1, None)
            x.val[tuple(index)] = 2
            print(0)


def load(x, kind = 'constant'):
    if kind in 'constant':
        if x.__len__() == 2:
            return x[0]*0 + 1
        elif x.__len__() == 3:
            return (x[0]**0) + (x[1]**0) + (30*x[2]**0)

    elif kind in 'continuous':
        if x.__len__() == 2:
            #  return (x[0]**4/(x[0]+2))+(x[1]**3/2)
            return (x[0]**4) + (x[1]**3)
        elif x.__len__() == 3:
            # return (x[0]**4/(x[0]+2))+(x[1]**3/2)+np.cos(x[2])
            return (x[0]**4) + (x[1]**3) + np.cos(x[2])

    elif kind in 'zero':
        return (x[0]*0 + 1)


    elif kind in 'manufactured':
        return (12*(x[0]**2)*x[1]**3) - (12*(x[0]**2)*x[1]) + (6*(x[0]**4)*x[1]) - 6*x[1]

    elif kind in 'discontinuous':
        val = np.zeros(tuple(x[0].shape))
        for x_i in np.ndindex(x[0].shape):

            if (abs(x[0][x_i]) < 0.2 or abs(x[1][x_i]) < 0.2) and (abs(x[0][x_i]) < 0.5 and abs(x[1][x_i]) < 0.7):
                val[x_i] = 10
                val[x_i] = 10

            else:
                val[x_i] = 1
                val[x_i] = 1

        return val




    else:
        raise ('Wrong kind of load')


def mat_fun(x, kind = 'constant'):
    if kind in 'continuous':
        val = np.zeros(tuple(2*(x.__len__(),)) + x[0].shape)
        # for x_i in np.ndindex(x[0].shape):

        val[0][0] = 1 + x[0]**2*x[1]**2*np.cos(x[0]**1) + 2
        val[1][1] = 1 + x[0]**2*x[1]**2*np.cos(x[1]**1) + 2

        return val

        #   return [[np.sin(x[0]**2) , x[1]* 0 , x[2]*0],
        #            [x[0]* 0 , 5*np.cos(5*x[1]) , x[2]* 0],
        #            [x[0]* 0 , x[1]* 0 , np.cos(x[2]**3)]]

    if kind in 'bilinear':

        val = np.zeros(tuple(2*(x.__len__(),)) + x[0].shape)
        for x_i in np.ndindex(x[0].shape):
            val[0][0][x_i] = x_i[0]**1
            val[1][1][x_i] = 2*x_i[1]**1

        return val

    elif kind in 'discontinuous':

        val = np.zeros(tuple(2*(x.__len__(),)) + x[0].shape)
        for x_i in np.ndindex(x[0].shape):

            if (abs(x[0][x_i]) < 0.3 or abs(x[1][x_i]) < 0.3) and (abs(x[0][x_i]) < 0.7 and abs(x[1][x_i]) < 0.7):
                val[0][0][x_i] = 1000
                val[1][1][x_i] = 1000

            else:
                val[0][0][x_i] = 1
                val[1][1][x_i] = 1

        return val

    elif kind in 'constant':
        val = np.zeros(tuple(2*(x.__len__(),)) + x[0].shape)
        for x_i in np.ndindex(x[0].shape):

            if (abs(x[0][x_i]) < 0.3 or abs(x[1][x_i]) < 0.3) and (abs(x[0][x_i]) < 0.7 and abs(x[1][x_i]) < 0.7):
                val[0][0][x_i] = 1
                val[1][1][x_i] = 1

            else:
                val[0][0][x_i] = 1
                val[1][1][x_i] = 1

        return val

    else:
        raise ('Wrong kind of material')


if __name__ == '__main__':

    dim = 2  # dim
    extreme = []
    maxres = []
    sol_GaNi_max = []
    sol_DG_max = []
    sol_col_max = []

    sol_GaNi_int = []
    sol_DG_int = []
    sol_col_int = []
    i = 0
    Ns = np.arange(4, 50, 2)  # [5,10,15,20,25]
    for n in Ns:  # np.arange(55, 37,3):
        N = np.array(dim*[n, ], dtype = np.int)

        # solWP, infoWP = problem_WP0(N, mat = 'constant', f = 'constant')

        sol_GaNi, info_GaNi = proble_Ga_Ni(N, mat = 'continuous', f = 'constant', useCC = True)
        sol_DG, info_DG = proble_Ga(N, mat = 'continuous', f = 'constant', useCC = True)

        sol_col, info_col = proble_Colocation(N, mat = 'continuous', f = 'constant', useCC = True)

        #     sol_non, info_non=proble_No0(N,mat='constant',f='constant',useCC=False)
        #     sol1_non, info1_non = proble_No1(N, mat = 'constant', f = 'constant',useCC=False)

        extreme.append(sol_GaNi.val.min())
        print('max')

        #     maxres.append(np.asarray(info['res_evol'])[-1])

        #        x0 = Chebyshev_element(name = 'x', N = N, shape = ())
        #        x0.set_nodal_coord()
        #        u_val = lambda x: ((x[0]**4)*x[1]**3) - x[1]**3 - ((x[0]**4)*x[1]) + x[1] if x.__len__() == 2 else (x[0]**4) + (x[1]**3) + np.cos(x[2])
        #        x0.val = u_val(x0.coord)

        sol_GaNi.set_nodal_coord()
        #  sol_GaNi.plot_val()
        sol_GaNi.integrate()

        sol_DG.set_nodal_coord()
        # sol_DG.plot_val()
        sol_DG.integrate()

        sol_col.set_nodal_coord()
        # sol_col.plot_val()
        sol_col.integrate()
        plt.show()

        sol_GaNi_int.append(sol_GaNi.integral)
        sol_DG_int.append(sol_DG.integral)
        sol_col_int.append(sol_col.integral)

        sol_GaNi_max.append(sol_GaNi.val.max())
        sol_DG_max.append(sol_DG.val.max())
        sol_col_max.append(sol_col.val.max())

        print(sol_GaNi_int[i])
        print(sol_DG_int[i])
        print(sol_col_int[i])
        i = i + 1
    #   diffcol = np.subtract(sol_GaNi.val, sol_DG.val)
    #   plot_field(sol_DG.coord, diffcol, dim)
    #      sol_non.set_nodal_coord()
    #       sol_non.plot_val()
    #        sol_non.integrate()

    #     sol1_non.set_nodal_coord()
    #     sol1_non.plot_val()
    #     sol1_non.integrate()

    #   plt.show()

    #   diff=np.subtract(sol1.val,sol.val)
    #    plot_field(sol.coord, diff, dim)
    #
    # print(sol.val.min())
    #        plt.show()
    # plot convegence
    sol_GaNi.plot_val()
    sol_DG.plot_val()
    sol_col.set_nodal_coord()

    plt.figure()
    plt.loglog(np.arange(0, np.asarray(info_GaNi['res_evol']).size), np.asarray(info_GaNi['res_evol']), label = "GaNi")
    plt.loglog(np.arange(0, np.asarray(info_DG['res_evol']).size), np.asarray(info_DG['res_evol']), label = "DG ")
    plt.loglog(np.arange(0, np.asarray(info_col['res_evol']).size), np.asarray(info_col['res_evol']),
               label = "collocation")
    xlabel = 'Number of iteration'
    ylabel = 'Norm of residuum'
    plt.title(N)
    #   plt.semilogy(np.arange(0, np.asarray(info_non['res_evol']).size), np.asarray(info_non['res_evol']),  label="2N_non")
    #       plt.semilogy(np.arange(0, np.asarray(info1_non['res_evol']).size), np.asarray(info1_non['res_evol']), label="N_non")
    plt.legend()
    # plt.show()

    #  print(sol_GaNi.val.max(), sol_GaNi.val.min())
    #   print(sol_GaNi.integral)
    #   print(sol_col.val.max(), sol_col.val.min())
    #   print(sol_col.integral)

    plt.figure()
    # plt.show()
    plt.plot(Ns, sol_GaNi_int, 'b--', label = 'GaNi', marker = 'o')
    plt.plot(Ns, sol_DG_int, '-.', label = 'DG', marker = 'x')
    plt.plot(Ns, sol_col_int, label = 'Coll', marker = 3)
    plt.title('Integral of solution')
    # xlabel = '[N,N]'
    plt.xlabel('[N,N]')
    plt.legend()

    plt.figure()
    plt.plot(Ns, sol_GaNi_max, 'b--', label = 'GaNi', marker = 'o')
    plt.plot(Ns, sol_DG_max, '-.', label = 'DG', marker = 'x')
    plt.plot(Ns, sol_col_max, label = 'Coll', marker = 2)
    #   plt.plot(Ns,np.full((Ns.size),  0.294684591104321),'D', label = 'matlab')
    plt.title('Max value of solution')
    # xlabel = 'Number of iteration'
    plt.xlabel('[N,N]')
    plt.legend()

    # print(info['norm_res'])
    # print(info_GaNi['kit'])
    # print(sol.val.min())

    # plt.show()
    #  plt.plot(np.arange(5,17,2), extreme)
    # plt.show()
    #  plt.plot(np.arange(0, 3), maxres)
    # plt.show()

    #  sol.set_nodal_coord()
    #   sol.plot_val()
    #   sol.integrate()
    #    plt.show()

    plt.show()
    print('End')
