import unittest
import itertools
import numpy as np
import copy

from Chebyshev.elements import Chebyshev_element
from Chebyshev.discrete_cheby import *
from Chebyshev.problem import load,mat_fun

from Chebyshev.plot_functions import plot_field
class Test_operators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dct(self):
        print('\nChecking discrete cosine transform...')
        for dim, Ni in itertools.product([2, 3],[2,4,8]):
            N=tuple(np.array(dim*[Ni, ], dtype = np.int))
            values = np.random.random_sample(N)
            Element=Chebyshev_element(name = 'x', N = N, shape = ())
            Element.val=values
            u1=copy.deepcopy( Element.val)

            Element.dct()
            Element.idct()

            u2=copy.deepcopy( Element.val)

            self.assertTrue(np.allclose(u1, u2, atol=1e-15,), msg = "dct does not preserve values for dim {} and  N{}".format(dim,Ni))

    def test_diracdelta(self): ## TODO: finnish for dD
        print('\nChecking discrete dirac-delta property...')
        for dim, N_n in itertools.product([1],[5,10,15]):
            for i in range(N_n):

                x = np.asarray( cheb_extrema_grid1D(N_n))

                Lx = np.zeros(len(x))
                for j in range(len(x)):
                    Lx[j] = tcheby_lagrange(x[j], i, N_n)

                self.assertTrue(np.isclose(Lx.sum(),1), msg = "Discrete dirac delta does not hold")

    def test_interpolate(self):
        print('\nChecking enlarge and decrease')
        for dim, Ni in itertools.product([2], [5,15,55]):
            N = np.array(dim*[Ni, ], dtype = np.int)
            dN = tuple(np.array(N*2 - 1, dtype = np.int))
            x0 = Chebyshev_element(name = 'x', N = N, shape = (),Fourier = False)
            x0.set_nodal_coord()
            x0.val = load(x0.coord,'continuous')

            a = copy.deepcopy(x0)
          #  print(x0.val)
            x0.interpolate()
          #  print(x0.val)
            x0.dct()
          #  print(x0.val)
            x0.shrink()
          #  print(x0.val)
            x0.idct()
        #    print(x0.val)
            self.assertTrue(np.allclose(a.val, x0.val, atol=1e-12,), msg = "Enlarge and shrink does not preserve values")

           # print('asshole')

    def test_integral(self):
        print('\nChecking integral')
        solution=[0,0.8,8.3317]
        for dim, Ni in itertools.product([2,3], [100,155]):
            N = np.array(dim*[Ni, ], dtype = np.int)
            dN = tuple(np.array(N*2 - 1, dtype = np.int))
            x0 = Chebyshev_element(name = 'x', N = N, shape = ())
            x0.set_nodal_coord()
            integrant = lambda x: (x[0]**4) + (x[1]**3) if x.__len__() == 2 else (x[0]**4) + (x[1]**3) + np.cos(x[2])
            x0.val = integrant(x0.coord)
            x0.integrate()
            self.assertTrue(np.isclose(x0.integral,solution[dim-1], rtol=1e-05, atol=1e-08, equal_nan=False), msg = "Integral does not work correctly")


    def test_integral_w_projection(self):
        print('\nChecking integral with projection')
        solution=[0,0.8,8.3317]
        for dim, Ni in itertools.product([2,3], [100,155]):
            N = np.array(dim*[Ni, ], dtype = np.int)
            dN = tuple(np.array(N*2 - 1, dtype = np.int))

            integrant = lambda x: (x[0]**4) + (x[1]**3) if x.__len__() == 2 else (x[0]**4) + (x[1]**3) + np.cos(x[2])
            u = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            u.set_nodal_coord()
            u.val = integrant(u.coord)

            u.integrate()
            i_1=u.integral
            u.dct()
            u.enlarge()
            u.idct()
            u.integrate()
            i_2=u.integral
            u.dct()
            u.shrink()
            u.idct()
            i_3 = u.integral

            self.assertTrue(np.isclose(i_1,i_2, rtol=1e-05, atol=1e-08, equal_nan=False), msg = "Integral does not work correctly")
            self.assertTrue(np.isclose(i_1, i_3, rtol = 1e-05, atol = 1e-08, equal_nan = False), msg = "Integral does not work correctly")

    def test_integral_with_projection(self):
        print('\nChecking preservation of integral during projections')
        for dim, Ni in itertools.product([2], [15,55,155]):
            N = np.array(dim*[Ni, ], dtype = np.int)

            u_val = lambda x: (x[0]**4) + (x[1]**3)#((x[0]**2)*x[1]**3) - x[0]**2*x[1] - x[1]**3 + x[1]





    def test_gradient(self): ## TODO : NOT FINISHED
        print('\nChecking gradient')
        #solution = [0, 0.8, 8.3317]

        for dim, Ni in itertools.product([2], [10]):
            N = np.array(dim*[Ni, ], dtype = np.int)
            print(N)
            fun_val = lambda x:     ((  x[0]**4) * x[1]**4) + ((  x[0]**3) *  x[1]**4)+((  x[0]**4) *  x[1]**3) -   x[1]**3 -  x[0]**4

            grad_fun_x = lambda x:  ((4*x[0]**3) * x[1]**4) + ((3*x[0]**2) *  x[1]**4)+((4*x[0]**3) *  x[1]**3) -            4*x[0]**3
            grad_fun_y = lambda x:  ((4*x[0]**4) * x[1]**3) + ((  x[0]**3) *4*x[1]**3)+((  x[0]**4) *3*x[1]**2) - 3*x[1]**2

            x0 = Chebyshev_element(name = 'x', N = N, shape = ())
            x0.set_nodal_coord()
            x0.val = fun_val(x0.coord)

            x0.dct_ortho()
            x0.grad_ortho()
            x0.idct_ortho()
          #  x0.plot_grad()

            control = Chebyshev_element(name = 'control', N = N, shape = (2,), multype = 'scal')
            control.set_nodal_coord()
            control.val[0] = grad_fun_x(control.coord)
            control.val[1] = grad_fun_y(control.coord)

         #   control.dct()


         #   control.plot_grad()
            plt.show()
            self.assertTrue(np.allclose(x0.val, control.val, rtol = 1e-010, atol = 1e-08, equal_nan = False),
                            msg = "Gradient does not work correctly")

        plt.show()

    def test_weighted_projection_BF(self):
        print('\nChecking weighted projection')
        coll_BF = []
        GaNi_BF = []
        adjoin_GaNi_BF = []
        adjoin_Ga_BF = []
        Ns = np.arange(4,10,1)#[6,10, 50, 100, 155, 200]#
        i = 0
        for dim, Ni in itertools.product([2], Ns):
            N = np.array(dim*[Ni, ], dtype = np.int)

           # u_val = lambda x: (((x[0]**2)*x[1]**3) - x[0]**2*x[1] - x[1]**3 + x[1])*np.cos(np.pi*x[0]/2)*np.cos(np.pi*x[1]/2)
            u_val = lambda x: (x[0]**2-1)*(x[1]**3-x[1]**1)#*np.cos(4*np.pi*x[0]/2)*np.cos(np.pi*x[1]*2)#*(-x[2]**2+1)
           # v_val = lambda x: (((x[0]**2)*x[1]**3) - x[0]**2*x[1] - x[1]**3 + x[1])*np.cos(np.pi*x[0]/2)*np.cos(np.pi*x[1]*2)
            v_val = lambda x: (x[0]**2 - 1)*(x[1]**3 - x[1]**1)#*np.cos(4*np.pi*x[0]/2)*np.cos(np.pi*x[1]*2)#*(-x[2]**2 + 1)
            u = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            u.set_nodal_coord()
            u.val = u_val(u.coord)

            material = Chebyshev_element(name = 'A_mat', N = N, shape = (), multype = '21')
            material.set_nodal_coord()
            material.val = load(material.coord, kind = 'constant') # (x[0]**4 ) + (x[1]**3)


            v = Chebyshev_element(name = 'v', N = N, shape = (), multype = 'scal')
            v.set_nodal_coord()
            v.val = v_val(v.coord)

            def collocation(a,b,mat):
                a.dct()
                a.idct()
                a.val=np.multiply(mat.val,a.val)
                a.dct()
                a.idct()

                original = Chebyshev_element(name = 'original', N = N, shape =(), multype = 'scal')
                original.val = np.multiply(a.val, b.val)
                original.integrate()
                return original.integral

            def GaNi(a, b,mat):
                a.dct()
                a.idct()
                a.val=np.multiply(mat.val,a.val)
                a.mul_by_cc_weights()
                b.dct()
                b.idct()

                both_side = Chebyshev_element(name = 'both_side', N = N, shape = (2,), multype = 'scal')
                both_side.val = np.multiply(a.val, b.val)
                #both_side.integrate()
                scalar = np.sum(both_side.val)
                return scalar#both_side.integral

            def adjoin_GaNi(a, b,mat):
                a.dct()
                a.idct()
                a.val = np.multiply(mat.val, a.val)
                a.mul_by_cc_weights()
                a.dct()
                a.idct()

                adjoin = Chebyshev_element(name = 'adjoin', N = N, shape = (), multype = 'scal')
                adjoin.val = np.multiply(a.val, b.val)
                scalar=np.sum(adjoin.val)
                return scalar

            def adjoin_Ga(a, b,mat):
                mat.dct()
                mat.enlarge()
                mat.idct()

                a.dct()
                a.enlarge()
                a.idct()
                a.val =(2**dim)*np.multiply(mat.val, a.val)
                a.mul_by_cc_weights()
                a.dct()
                a.shrink()
                a.idct()
                adjoin = Chebyshev_element(name = 'adjoin', N = N, shape = (), multype = 'scal')
                adjoin.val = np.multiply(a.val, b.val)

                scalar=np.sum(adjoin.val)
                return scalar

            print("solutions for N={}".format(N))
            coll_BF.append(collocation(copy.deepcopy(u),copy.deepcopy(v),copy.deepcopy(material)))
            print('  sum((-FFAFFu,v).*W={})'.format( coll_BF[i]))

            GaNi_BF.append(GaNi(copy.deepcopy(u), copy.deepcopy(v),copy.deepcopy(material)))
            print('  sum((WAFFu,FFv)   ={})'.format(GaNi_BF[i]))

            adjoin_GaNi_BF.append(adjoin_GaNi(copy.deepcopy(u), copy.deepcopy(v),copy.deepcopy(material)))
            print('  sum((FFWAFFu,v)   ={})'.format(adjoin_GaNi_BF[i]))

            adjoin_Ga_BF .append(adjoin_Ga(copy.deepcopy(u), copy.deepcopy(v),copy.deepcopy(material)))
            print('  sum((FPFWAFPFu,v) ={})'.format(adjoin_Ga_BF[i]))
            i=i+1

       # plt.plot(Ns, coll_BF, label = "Collocation")
       # plt.plot(Ns, GaNi_BF, label = " GaNi_BF")
      #  plt.plot(Ns, adjoin_GaNi_BF, label = "adjoin_GaNi_BF")
      #  plt.plot(Ns, adjoin_Ga_BF, label = "adjoin_Ga_BF")
       # plt.legend()
       # plt.show()

    """          self.assertTrue(np.isclose(GaNi_BF, coll_BF, rtol = 1e-05, atol = 1e-08, equal_nan = False),
                            msg = "Weighted projection does not work correctly")
            self.assertTrue(np.isclose(adjoin_GaNi_BF, adjoin_Ga_BF, rtol = 1e-05, atol = 1e-08, equal_nan = False),
                            msg = "Weighted projection does not work correctly")
            self.assertTrue(np.isclose(adjoin_GaNi_BF, coll_BF, rtol = 1e-05, atol = 1e-08, equal_nan = False),
                            msg = "Weighted projection does not work correctly")  
    """

    def test_elliptic_BF(self):
        print('\nChecking elliptic bilinear form')
        coll_BF = []
        GaNi_BF = []
        adjoin_GaNi_BF = []
        adjoin_Ga_BF = []
        Ns = np.arange(2,12,1)#[10, 50, 100, 155, 200]
        i = 0
        for dim, Ni in itertools.product([2], Ns):

            N = np.array(dim*[Ni, ], dtype = np.int)
            dN = tuple(np.array(N*2 - 1, dtype = np.int))
            
           #  (((x[0]**2)*x[1]**3) - x[0]**2*x[1] - x[1]**3 + x[1])#*np.cos(np.pi*x[0]/2)*np.cos(np.pi*x[1]/2)
            u_val = lambda x: (x[0]**2 - 1)*(x[1]**3 - x[1]**1)
            v_val = lambda x: (x[0]**2 - 1)*(x[1]**3 - x[1]**1)       # (((x[0]**2)*x[1]**3) - x[0]**2*x[1] - x[1]**3 + x[1])#*np.cos(np.pi*x[0]/2)*np.cos(np.pi*x[1]*2)
            u = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            u.set_nodal_coord()
            u.val = u_val(u.coord)

            v = Chebyshev_element(name = 'v', N = N, shape = (), multype = 'scal')
            v.set_nodal_coord()
            v.val = v_val(v.coord)

            material = Chebyshev_element(name = 'A_mat', N = N, shape = 2*[dim], multype = '21')
            material.set_nodal_coord()
            material.val = mat_fun(material.coord, kind = 'constant')

            materialDG = Chebyshev_element(name = 'A_mat', N = dN, shape = 2*[dim], multype = '21')
            materialDG.set_nodal_coord()
            materialDG.val = mat_fun(materialDG.coord, kind = 'constant')
           #materialDG.mul_by_cc_weights()

            def collocation(a,b,mat):
                a.dct()
                a.grad()
                a.idct()
                a.val = (mat*a).val
                a.dct()
                a.div()

                original = Chebyshev_element(name = 'original', N = N, shape =(), multype = 'scal')
                original.val = np.multiply(-a.val, b.val)

                original.integrate()
                return original.integral

            def GaNi(a, b,mat):
                a.dct()
                a.grad()
                a.idct()
                a.val = (mat*a).val
                a.mul_by_cc_weights()

                b.dct()
                b.grad()
                b.idct()

                both_side = Chebyshev_element(name = 'both_side', N = N, shape = (2,), multype = 'scal')
                both_side.val = np.multiply(a.val, b.val)
             #   print('ab')
             #   print(both_side.val)
                #both_side.integrate()
                scalar = np.sum(both_side.val)
                return scalar # both_side.integral

            def adjoin_GaNi(a, b,mat):
                a.dct()
                a.grad()
                a.idct()

                a.val = (mat*a).val
                a.mul_by_cc_weights()

                a.idct_trans()
                a.grad_transpose()
                a.dct_trans()

                adjoin = Chebyshev_element(name = 'adjoin', N = N, shape = (), multype = 'scal')
                adjoin.val = np.multiply(a.val, b.val)
                scalar=np.sum(adjoin.val)
                return scalar

            def adjoin_Ga(a, b,mat):
                a.dct()
                a.grad()
                a.enlarge()
                a.idct()
                mat.mul_by_cc_weights()
                a.val = (mat*a).val
                #a.mul_by_cc_weights()

                a.idct_trans()
                a.shrink()
                a.grad_transpose()
                a.dct_trans()

                adjoin = Chebyshev_element(name = 'adjoin', N = N, shape = (), multype = 'scal')
                adjoin.val = np.multiply(a.val, b.val)
                #adjoin.integrate()
                scalar=np.sum(adjoin.val)
                return scalar

            print("solutions for N={}".format(N))
            coll_BF.append(collocation(copy.deepcopy(u),copy.deepcopy(v),copy.deepcopy(material)))
            print('  sum((-FDFAFGFu,v).*W={})'.format( coll_BF[i]))

            GaNi_BF.append(GaNi(copy.deepcopy(u), copy.deepcopy(v),copy.deepcopy(material)))
            print('   sum((WAFGFu,FGFv)  ={})'.format(GaNi_BF[i]))

            adjoin_GaNi_BF.append( adjoin_GaNi(copy.deepcopy(u), copy.deepcopy(v),copy.deepcopy(material)))
            print('  sum((FGtFWAFGFu,v)  ={})'.format(adjoin_GaNi_BF[i]))

            adjoin_Ga_BF.append(adjoin_Ga(copy.deepcopy(u), copy.deepcopy(v),copy.deepcopy(materialDG)))
            print('sum((FGtPFWAFPGFu,v)  ={})'.format(adjoin_Ga_BF[i]))

            i=i+1

        plt.plot(Ns, coll_BF, label = "Collocation")
        plt.plot(Ns, GaNi_BF, label = " GaNi_BF")
        plt.plot(Ns, adjoin_GaNi_BF, label = "adjoin_GaNi_BF")
        plt.plot(Ns, adjoin_Ga_BF, label = "adjoin_Ga_BF")
        plt.legend()
        plt.show()

            #print(GaNi_BF)
            #print(coll_BF)

            #print("solutions")

            #self.assertTrue(np.isclose(GaNi_BF, coll_BF, rtol = 1e-05, atol = 1e-08, equal_nan = False),
             #               msg = "Adjoin operator does not work correctly")

    def test_adjoin_operator(self):
        print('\nChecking adjoin_operator form')
        coll_BF = []
        GaNi_BF = []
        adjoin_GaNi_BF = []
        adjoin_GaNi_T = []
        adjoin_Ga_BF = []
        testers= []
        Ns = np.arange(2,12, 1)  # [10, 50, 100, 155, 200]
        i = 0
        for dim, Ni in itertools.product([2], Ns):
            N = np.array(dim*[Ni, ], dtype = np.int)
            dN = tuple(np.array(N*2 - 1, dtype = np.int))
            print(N)
            #  (((x[0]**2)*x[1]**3) - x[0]**2*x[1] - x[1]**3 + x[1])#*np.cos(np.pi*x[0]/2)*np.cos(np.pi*x[1]/2)
            u_val1 = lambda x:  x[0]**2 #-x[0]**3 # 4*x[0]**3 -x[0]#(x[0]**2 -x[1]**3 - 3)
            u_val2 = lambda x:  x[0]*0  #  x[1]**2 -x[0]**3 #  3*x[1]**2 -x[0]**3

            v_val =  lambda x:  (x[0]**3-x[0])*(x[0]**3-x[0]) #(x[0]**2 - 1)*(x[1]**3 - x[1]**1)+x[0]**3

            exp_val = lambda x: 3*x[0]

                  # (((x[0]**2)*x[1]**3) - x[0]**2*x[1] - x[1]**3 + x[1])#*np.cos(np.pi*x[0]/2)*np.cos(np.pi*x[1]*2)
            u = Chebyshev_element(name = 'u', N = N, shape = (2,), multype = 'scal')
            u.set_nodal_coord()
            u.val[0] = u_val1(u.coord)
            u.val[1] = u_val2(u.coord)

            v = Chebyshev_element(name = 'v', N = N, shape = (), multype = 'scal')
            v.set_nodal_coord()
            v.val = v_val(v.coord)

            exp = Chebyshev_element(name = 'exp', N = N, shape = (), multype = 'scal')
            exp.set_nodal_coord()
            exp.val = exp_val(exp.coord)


            def GaNi(a, b):

                b.dct()

                b.grad()
               # print(b.val)
                b.idct()

                both_side = Chebyshev_element(name = 'both_side', N = N, shape = (2,), multype = 'scal')
                both_side.val = np.multiply(a.val, b.val)
               # both_side.dct()
               # print(both_side.val)
               # both_side.idct()
                #   print('ab')
                #   print(both_side.val)
                both_side.integrate()
                #scalar = np.sum(both_side.val)
                return both_side.integral# scalar  #

            def coll(a, b):
                #  print('coll a dct')
             #   print(a.val)
                a.dct()
              #  print(a.val)
                a.div()
            #    print(a.val)
                #a.idct()

                both_side = Chebyshev_element(name = 'both_side', N = N, shape = (2,), multype = 'scal')
                both_side.val = np.multiply(a.val, b.val)
                #   print('ab')
                #   print(both_side.val)
                both_side.integrate()
                #scalar = np.sum(both_side.val)
                return both_side.integral# scalar  #



            def tester(a, b):
                a.dct()
              #  print(a.val)
                a.idct()
                adjoin = Chebyshev_element(name = 'test', N = N, shape = (), multype = 'scal')
                adjoin.val = np.multiply(a.val, b.val)
                adjoin.dct()
               # print(adjoin.val)

                adjoin.idct()
                #    print('ab')

                #   print(adjoin.val)
                adjoin.integrate()
                #scalar = np.sum(adjoin.val)
                return adjoin.integral#scalar  # #


            def adjoin_GaNi(a, b):
                print('Gani')
                a.dct()

            #    print(a.val)
                a.grad_transpose()

             #   print(a.val)
                a.idct()

                adjoin = Chebyshev_element(name = 'adjoin', N = N, shape = (), multype = 'scal')
                adjoin.val = np.multiply(a.val, b.val)
                adjoin.dct()
           #     print(adjoin.val)

                adjoin.idct()
                #    print('ab')

                #   print(adjoin.val)
                adjoin.integrate()
                #scalar = np.sum(adjoin.val)
                return adjoin.integral#scalar  # #

            def adjoin_GaNi_Tran(a, b, ):
              #  print('Gani')
              #  print(a.val)
                a.Fourier = True
                a.idct_trans()
              #  print(a.val)
           #     iFa=a.idct()
            #    print(iFa.val)
                a.Fourier = True
                a.grad_transpose()
                #      print(a.val)
                #       print(a.val)
                a.Fourier=False
                a.dct_trans()
             #   print(a.val)
                # b.idct()
                # plt.show()
                a.Fourier = False
                adjoin = Chebyshev_element(name = 'adjoin', N = N, shape = (), multype = 'scal')
                adjoin.val = np.multiply(a.val, b.val)
                #    print('ab')
                #   print(adjoin.val)
                adjoin.integrate()
                # scalar = np.sum(adjoin.val)
                return adjoin.integral  # scalar  # #





            coll_BF.append(coll(copy.deepcopy(u), copy.deepcopy(v)))
            print('   sum((FDFu,v)  ={})'.format(coll_BF[i]))

            testers.append(tester(copy.deepcopy(exp), copy.deepcopy(v)))
            print('   sum((exp,v)  ={})'.format(testers[i]))

            # print('  sum((-FDFAFGFu,v).*W={})'.format(coll_BF[i]/(N[0]*N[1])))
            GaNi_BF.append(GaNi(copy.deepcopy(u), copy.deepcopy(v)))
            print('   sum((u,FGFv)  ={})'.format(GaNi_BF[i]))
            # print('   sum((WAFGFu,FGFv)  ={})'.format(GaNi_BF[i]/(N[0]*N[1])))
            adjoin_GaNi_BF.append(adjoin_GaNi(copy.deepcopy(u), copy.deepcopy(v)))
            print('  sum((FGtFiu,v)  ={})'.format(adjoin_GaNi_BF[i]))
            #   print('  sum((FGtFWAFGFu,v)  ={})'.format(adjoin_GaNi_BF[i]/(N[0]*N[1])))
            adjoin_GaNi_T.append(adjoin_GaNi_Tran(copy.deepcopy(u), copy.deepcopy(v)))
            print('  sum((FitGtFtu,v)  ={})'.format(adjoin_GaNi_T[i]))


            i = i + 1

        plt.plot(Ns, coll_BF, label = "Collocation")
        plt.plot(Ns, GaNi_BF, label = " GaNi_BF")
        plt.plot(Ns, adjoin_GaNi_BF, label = "adjoin_GaNi_BF")
        plt.plot(Ns, adjoin_GaNi_T, label = "adjoin_GaNi_T")
        plt.legend()
        plt.show()

    def test_grad_transpose(self):
        print('\nChecking adjoin to gradient')
        Ns = np.arange(4, 25, 1)
        for dim, Ni in itertools.product([2], Ns):
            N = np.array(dim*[Ni, ], dtype = np.int)
            print(N)
           # u_val = lambda x: (x[0]**0 )*(x[1]**0 )

            u = Chebyshev_element(name = 'u', N = N, shape = (2,), multype = 'scal')
            u.set_nodal_coord()
            u.val[0] = np.random.rand(N[0], N[1])*1
            u.val[1] = np.random.rand(N[0], N[1])*1
            u.Fourier = True
            u.idct()

           # u.Fourier = True


            v = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            v.set_nodal_coord()
            v.val = np.random.rand(N[0], N[1])*1#u_val(u.coord)
            v.Fourier = True
            v.idct()

            #v.Fourier = True
            # u.grad_transpose()
            # print(u.val)

            def grad_forward(a, b):
                b.dct()
                b.grad()
                b.idct()

                scal = Chebyshev_element(name = 'scal', N = N, shape = (), multype = 'scal')
                scal.set_nodal_coord()
                scal.val = np.multiply(a.val, b.val)
                return scal.val

            def transposed_grad(a, b):
               # a.Fourier = True
                a.idct_trans()
             #   a.Fourier = True
                a.grad_transpose()
              #  a.Fourier = False
                a.dct_trans()
                scal2 = Chebyshev_element(name = 'scal2', N = N, shape = (), multype = 'scal')
                scal2.set_nodal_coord()
                scal2.val = np.multiply(a.val, b.val)
                return scal2.val


            forward = grad_forward(copy.deepcopy(u), copy.deepcopy(v))
            #print(forward)
            transposed = transposed_grad(copy.deepcopy(u), copy.deepcopy(v))
            #print(transposed)
            #print('  sum((u,iFGFv)   ={})'.format(np.sum(forward)))
            #print('  sum((FtGtiFtu,v)   ={})'.format(np.sum(transposed)))
        #    print(' dif sum((Ftu,v)-(u,Fv))   ={})'.format(np.sum(transposed)-np.sum(forward)))

            self.assertTrue(np.isclose(np.sum(forward), np.sum(transposed), rtol = 1e-10, atol = 1e-08, equal_nan = False),
                    msg = "Transposed DCT does not work correctly")

    def test_trans_dct(self):
        print('\nChecking transposed DCT')
        Ns = np.arange(4,15, 1)
        for dim, Ni in itertools.product([2], Ns):
            N = np.array(dim*[Ni, ], dtype = np.int)
            print(N)
            u = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            u.set_nodal_coord()
            u.val =np.random.rand(N[0],N[1])*1
            u.Fourier=True
            u.idct()

            v = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            v.set_nodal_coord()
            v.val = np.random.rand(N[0],N[1])*1
            v.Fourier = True
            v.idct()

            def transform(a, b):
                b.dct()
                b.idct()
                scal = Chebyshev_element(name = 'scal', N = N, shape = (), multype = 'scal')
                scal.set_nodal_coord()
                scal.val = np.multiply(a.val, b.val)
                return  scal.val

            def transposed_transform(a, b):
               # a.Fourier=True
                a.idct_trans()
                a.dct_trans()
                scal2 = Chebyshev_element(name = 'scal2', N = N, shape = (), multype = 'scal')
                scal2.set_nodal_coord()
                scal2.val = np.multiply(a.val, b.val)
                return  scal2.val

            forward=transform(copy.deepcopy(u),copy.deepcopy(v))
            transposed = transposed_transform(copy.deepcopy(u), copy.deepcopy(v))

          #  print('  sum((u,Fv)   ={})'.format(np.sum(forward)))
          #  print('  sum((Ftu,v)   ={})'.format(np.sum(transposed)))
          #  print(' dif sum((Ftu,v)-(u,Fv))   ={})'.format(np.sum(transposed-forward)))

            self.assertTrue(np.isclose(np.sum(forward), np.sum(transposed), rtol = 1e-10, atol = 1e-08, equal_nan = False),
                           msg = "Transposed DCT does not work correctly")

    def test_trans_idct(self):
        print('\nChecking transposed DCT')
        Ns = np.arange(2, 15, 1)
        for dim, Ni in itertools.product([2], Ns):
            N = np.array(dim*[Ni, ], dtype = np.int)
            print(N)
            u = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            u.set_nodal_coord()
            u.val = np.random.rand(N[0], N[1])*1
            u.Fourier = True
            u.idct()

            v = Chebyshev_element(name = 'u', N = N, shape = (), multype = 'scal')
            v.set_nodal_coord()
            v.val = np.random.rand(N[0], N[1])*1
            v.Fourier = True
            v.idct()

            def transform(a, b):
                b.Fourier=True
                b.idct()
                scal = Chebyshev_element(name = 'scal', N = N, shape = (), multype = 'scal')
                scal.set_nodal_coord()
                scal.val = np.multiply(a.val, b.val)
                return scal.val

            def transposed_transform(a, b):
              #  a.Fourier = True
                a.idct_trans()
                scal2 = Chebyshev_element(name = 'scal2', N = N, shape = (), multype = 'scal')
                scal2.set_nodal_coord()
                scal2.val = np.multiply(a.val, b.val)
                return scal2.val

            forward = transform(copy.deepcopy(u), copy.deepcopy(v))
            transposed = transposed_transform(copy.deepcopy(u), copy.deepcopy(v))

        #    print('  sum((u,iFv)   ={})'.format(np.sum(forward)))
         #   print('  sum((iFtu,v)   ={})'.format(np.sum(transposed)))
          #  print(' dif sum((iFtu,v)-(u,iFv))   ={})'.format(np.sum(transposed - forward)))

            self.assertTrue(np.isclose(np.sum(forward), np.sum(transposed), rtol = 1e-10, atol = 1e-08, equal_nan = False),
               msg = "Transposed iDCT does not work correctly")

if __name__=="__main__":
   # unittest.main()
    unittest.main(Test_operators().test_elliptic_BF())
  #  unittest.main(Test_operators().test_grad_transpose())
 #   unittest.main(Test_operators().test_elliptic_BF(),Test_operators().test_trans_idct(),Test_operators().test_trans_dct(),Test_operators().test_grad_transpose())
 #   unittest.main(Test_operators().test_trans_dct())
#  unittest.main(Test_operators().test_adjoin_operator())
   # unittest.main(Test_operators().test_gradient())
   # unittest.main(Test_operators().test_weighted_projection_BF())
  #  unittest.main(Test_operators().test_dct())

   #

#
  # unittest.main(Test_operators().test_elliptic_BF())
#   # show how test works
  #




   #
    #
  #  unittest.main(Test_operators().test_elliptic_BF())
   # unittest.main(Test_operators().test_gradient(),Test_operators().test_weighted_projection_BF(),Test_operators().test_elliptic_BF())

   # unittest.main(Test_operators().test_weighted_projection_BF())

