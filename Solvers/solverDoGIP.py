# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:49:44 2019

@author: rajuc
"""

import numpy as np
from Functions import functions
from PreProcessing import quad as quad
import itertools
from PreProcessing.polyNd import polyNd
import scipy.sparse as scsp
import scipy.sparse.linalg as linalg


class SolverDoGIP(object):

    def __init__(self, grid, problem_type=2):
        self.grid=grid
        self.problemType=problem_type

    def set_numQuadPts_for_bilinear(self, numQuadPts):

        self.numQuadPts=np.asarray(numQuadPts)

        self.points, self.weights=quad.Quadrature.get_gauss_quad(numQuadPts)

        if self.problemType==1 or self.problemType==2:

            self.basisAtQuadPts=np.empty(self.grid.order+1, np)
            self.basisDoubleGridAtQuadPts=np.empty(self.grid.referenceElement.order_double_grid+1, np)

            for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
                self.basisAtQuadPts[I]=np.zeros(numQuadPts)

            for I in itertools.product(*[range(n) for n in (self.grid.referenceElement.order_double_grid+1)]):
                self.basisDoubleGridAtQuadPts[I]=np.zeros(numQuadPts)

            basis=self.grid.referenceElement.nodal_basis
            basisDoubleGrid=self.grid.referenceElement.nodal_basis_double_grid

            for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
                for J in itertools.product(*[range(n) for n in (numQuadPts)]):
                    self.basisAtQuadPts[I][J]=basis[I].value(self.points[J])

            for I in itertools.product(*[range(n) for n in (self.grid.referenceElement.order_double_grid+1)]):
                for J in itertools.product(*[range(n) for n in (numQuadPts)]):
                    self.basisDoubleGridAtQuadPts[I][J]=basisDoubleGrid[I].value(self.points[J])

        if self.problemType==2:
            self.basisGradAtQuadPts=np.empty(self.grid.order+1, np)

            for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
                self.basisGradAtQuadPts[I]=np.zeros(np.hstack((numQuadPts, self.grid.domain.dimension)))

            basisGrad=self.grid.referenceElement.nodal_basis_gradient

            for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
                for J in itertools.product(*[range(n) for n in (numQuadPts)]):
                    self.basisGradAtQuadPts[I][J]=np.asarray(polyNd.get_value_for_polyNdarray(basisGrad[I], self.points[J]))

    def set_numQuadPts_for_linear(self, numQuadPts):
        pass

    def build_A_DoGIP(self):

        for I in itertools.product(*[range(n) for n in (self.grid.numElements)]):
            self._build_A_T_DoGIP(I)

    def _build_A_T_DoGIP(self, elementID):
        n=np.prod(self.grid.referenceElement.numNodes_double_grid)
        a_diag=np.zeros(n)
        numQuadPts=self.weights.shape
        element=self.grid.Elements[elementID]
        detJ=np.linalg.det(element.A)
        A_inv=element.A_inv

        if self.problemType==1:

            for K in itertools.product(*[range(n) for n in numQuadPts]):
                globalCoord=np.dot(element.A, self.points[K])+element.b
                M=functions.material_scalar_value(globalCoord)
                for i in range(n):
                    I=self.grid.referenceElement.get_local_nodeID_double_grid(i)
                    a_diag[i]+=M*self.basisDoubleGridAtQuadPts[tuple(I)][K]*self.weights[K]*detJ

            element.A_T_DoGIP=scsp.spdiags(a_diag, 0, n, n)

        elif self.problemType==2:

            element.A_T_DoGIP=np.zeros((self.grid.domain.dimension, self.grid.domain.dimension, n))

            for K in itertools.product(*[range(n) for n in numQuadPts]):
                globalCoord=np.dot(element.A, self.points[K])+element.b
                M=functions.material_matrix_value(globalCoord)
                for i in range(n):
                    I=self.grid.referenceElement.get_local_nodeID_double_grid(i)
                    for r in range(self.grid.domain.dimension):
                        for s in range(self.grid.domain.dimension):
                            element.A_T_DoGIP[r, s, i]+=M[r, s]*A_inv[r, r]*A_inv[s, s]*self.basisDoubleGridAtQuadPts[tuple(I)][K]*detJ

        self.grid.Elements[elementID]=element

    def build_interpolation_operators(self):
        numNodes_double_grid=self.grid.referenceElement.numNodes_double_grid
        numNodes=self.grid.referenceElement.numNodes
        basis=self.grid.referenceElement.nodal_basis

        if self.problemType==1:
            B=np.zeros((np.prod(numNodes_double_grid), np.prod(numNodes)))

            i=0
            for I in itertools.product(*[range(n) for n in numNodes]):
                S=tuple(I[::-1])
                j=0
                for J in itertools.product(*[range(n) for n in numNodes_double_grid]):
                    T=tuple(J[::-1])
                    B[j, i]=basis[S].value(self.grid.referenceElement.coord_double_grid[T])
                    j+=1
                i+=1

            self.grid.referenceElement.B=scsp.csr_matrix(B)
            self.grid.referenceElement.B_adj=scsp.csr_matrix(B.T)

        elif self.problemType==2:

            basisGrad=self.grid.referenceElement.nodal_basis_gradient
            B=np.zeros((self.grid.domain.dimension, np.prod(numNodes_double_grid), np.prod(numNodes)))

            i=0
            for I in itertools.product(*[range(n) for n in numNodes]):
                S=tuple(I[::-1])
                j=0
                for J in itertools.product(*[range(n) for n in numNodes_double_grid]):
                    T=tuple(J[::-1])
                    B[:, j, i]=polyNd.get_value_for_polyNdarray(basisGrad[S], self.grid.referenceElement.coord_double_grid[T])
                    j+=1
                i+=1

            for dim in range(self.grid.domain.dimension):
                self.grid.referenceElement.B.append(scsp.csr_matrix(B[dim, :, :]))

            for k in range(np.prod(numNodes)):
                self.grid.referenceElement.B_adj.append(scsp.csr_matrix(B[:, :, k]))

    def operate_with_B(self, u):
        num_nodes_double_grid=np.prod(self.grid.referenceElement.numNodes_double_grid)

        if self.problemType==1:
            v=self.grid.referenceElement.B.dot(u)

        elif self.problemType==2:
            v=np.zeros((self.grid.domain.dimension, num_nodes_double_grid))
            for dim in range(self.grid.domain.dimension):
                B=self.grid.referenceElement.B[dim]
                v[dim, :]=B.dot(u)

        return v

    def operate_with_B_Adjoint(self, v):
        num_nodes=np.prod(self.grid.referenceElement.numNodes)

        u=np.zeros(num_nodes)

        if self.problemType==1:
            u=self.grid.referenceElement.B_adj.dot(v)

        elif self.problemType==2:
            for k in range(num_nodes):
                u[k]=(self.grid.referenceElement.B_adj[k].multiply(v)).sum()

        return u

    def compute_A_T_DoGIP_mult_Bv_T(self, elementID, Bv_T):
        A_T=self.grid.Elements[tuple(elementID)].A_T_DoGIP
        result=np.zeros(Bv_T.shape)

        if self.problemType==1:
            result=A_T.dot(Bv_T)

        elif self.problemType==2:
            for i in range(Bv_T.shape[1]):
                result[:, i]=A_T[:, :, i].dot(Bv_T[:, i])

        return result

    def build_global_source_vector(self):
        for I in itertools.product(*[range(n) for n in self.grid.numElements]):
            globalNodes=self.grid.Elements[I].globalNodeNum
            self.build_element_source_vector(I)
            for i in range(len(globalNodes)):
                self.grid.global_source_vector[globalNodes[i]]+=self.grid.Elements[I].source_vector[i]

    def build_element_source_vector(self, elementID):
        element=self.grid.Elements[elementID]
        detJ=np.linalg.det(element.A)
        nNodes=np.prod(element.numNodes)
        for i in range(nNodes):
            I=element.get_local_nodeID(i)
            integral=0.
            for K in itertools.product(*[range(n) for n in self.numQuadPts]):
                globalCoord=np.dot(element.A, self.points[K])+element.b
                f=functions.source_function_value(globalCoord)
                integral=integral+f*self.basisAtQuadPts[tuple(I)][K]*self.weights[K]*detJ
            element.source_vector[i]=integral

        self.grid.Elements[elementID]=element

    def impose_dirichlet_BC(self, dirichlet_bc):

        self.dirichletBC=dirichlet_bc
        self.homoDirichletNodeList=[]
        self.homoDirichletBCValue=[]
        self.inHomoDirichletNodeList=[]
        self.inHomoDirichletBCValue=[]

        self.unKnownDOF=list(range(np.prod(self.grid.numNodes)))

        for bc in range(len(self.dirichletBC)):
            nodelist=self.grid.get_nodes_on_boundary(self.dirichletBC[bc]['boundary'])
            nodelist=np.reshape(nodelist, np.prod(nodelist.shape))

            for i in range(len(nodelist)):
                nodeNum=self.grid.get_global_node_num(nodelist[i])
                for j in range(len(self.grid.periodicNodes)):
                    if self.grid.periodicNodes[j][1]==nodeNum:
                        nodeNum=self.grid.periodicNodes[j][0]

                nodeID=self.grid.get_global_node_ID(nodeNum)
                bcValue=self.dirichletBC[bc]['value'](self.grid.coord[tuple(nodeID)])

                if bcValue<1e-6:
                    self.homoDirichletNodeList.append(nodeNum)
                    self.homoDirichletBCValue.append(bcValue)
                else:
                    self.inHomoDirichletNodeList.append(nodeNum)
                    self.inHomoDirichletBCValue.append(bcValue)
                    self.grid.solution_array_g[nodeNum]=bcValue

                self.unKnownDOF.remove(nodeNum)

        self.numDirichletNodes=len(self.homoDirichletNodeList)+len(self.inHomoDirichletNodeList)

    def full_matrix_vector_product(self, v):

        v_out=np.zeros(v.shape)

        for I in itertools.product(*[range(n) for n in self.grid.numElements]):
            v_T=v[list(self.grid.Elements[I].globalNodeNum)]
            Bv_T=self.operate_with_B(v_T)
            A_T_mult_Bv_T=self.compute_A_T_DoGIP_mult_Bv_T(I, Bv_T)
            v_out[list(self.grid.Elements[I].globalNodeNum)]+=self.operate_with_B_Adjoint(A_T_mult_Bv_T)

        return v_out

    def matrix_vector_product_krylov(self, v):

        v_out=np.zeros(self.grid.solution_array.shape)
        v[self.homoDirichletNodeList]=0.
        v[self.inHomoDirichletNodeList]=0.
        # v_out[self.unKnownDOF] = v
        # print((self.grid.global_stiff_matrix.dot(v_out))[self.unKnownDOF].shape)
        for I in itertools.product(*[range(n) for n in self.grid.numElements]):
            v_T=v[list(self.grid.Elements[I].globalNodeNum)]
            Bv_T=self.operate_with_B(v_T)
            A_T_mult_Bv_T=self.compute_A_T_DoGIP_mult_Bv_T(I, Bv_T)
            v_out[list(self.grid.Elements[I].globalNodeNum)]+=self.operate_with_B_Adjoint(A_T_mult_Bv_T)

        v_out[self.homoDirichletNodeList]=0.
        v_out[self.inHomoDirichletNodeList]=0.
        # return (self.grid.global_stiff_matrix.dot(v_out))[self.unKnownDOF]
        return v_out

    def solve_linear_system(self):
        self.grid.global_source_vector-=self.full_matrix_vector_product(self.grid.solution_array_g)
        self.grid.global_source_vector[self.homoDirichletNodeList]=0.
        self.grid.global_source_vector[self.inHomoDirichletNodeList]=0.
        self.A=linalg.LinearOperator(shape=(np.prod(self.grid.numNodes), np.prod(self.grid.numNodes)), dtype=np.float64, matvec=self.matrix_vector_product_krylov)
        # self.A = linalg.LinearOperator(shape = (len(self.unKnownDOF), len(self.unKnownDOF)), matvec = self.matrix_vector_product_krylov)
        self.grid.solution_array, self.info=linalg.gmres(self.A, self.grid.global_source_vector, tol=1e-6)

        self.grid.solution_array+=self.grid.solution_array_g

        for i in range(len(self.grid.periodicNodes)):
            self.grid.solution_array[self.grid.periodicNodes[i][1]]=self.grid.solution_array[0][self.grid.periodicNodes[i][0]]

    def get_interpolation_operators(self):
        return self.grid.referenceElement.B

#    @staticmethod
#    def compute_interpolation_matrix(ref_element, problemID):
#
#        nodes_double_grid = np.prod(ref_element.numNodes_double_grid)
#        nodes = np.prod(ref_element.numNodes)
#        basis = ref_element.nodal_basis
#
#        if problemID == 1:
#            B_hat = scsp.lil_matrix((nodes_double_grid, nodes))
#
#        if problemID == 1:
#            i = 0
#            for I in itertools.product(*[range(n) for n in ref_element.numNodes]):
#                S = tuple(I[::-1])
#                j = 0
#                for J in itertools.product(*[range(n) for n in ref_element.numNodes_double_grid]):
#                    T = tuple(J[::-1])
#                    B_hat[j,i] = basis[S].value(ref_element.Coord_double_grid[T])
#                    j += 1
#                i += 1
#
#        B_hat = B_hat.tocsr()
#        return B_hat
