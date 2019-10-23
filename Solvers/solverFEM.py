# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:28:00 2019

@author: rajuc
"""
import numpy as np
from Functions import functions
from PreProcessing import quad as quad
import itertools
from PreProcessing.polyNd import polyNd
import scipy.sparse as scsp
import scipy.sparse.linalg as linalg


class SolverFEM(object):

    def __init__(self, grid, problem_type=2):
        self.grid=grid
        self.problemType=problem_type

    def set_numQuadPts_for_bilinear(self, numQuadPts):
        self.numQuadPts=np.asarray(numQuadPts)

        self.points, self.weights=quad.Quadrature.get_gauss_quad(numQuadPts)

        if self.problemType==1 or self.problemType==2:

            self.basisAtQuadPts=np.empty(self.grid.order+1, np)
            for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
                self.basisAtQuadPts[I]=np.zeros(numQuadPts)

            basis=self.grid.referenceElement.nodal_basis

            for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
                for J in itertools.product(*[range(n) for n in (numQuadPts)]):
                    self.basisAtQuadPts[I][J]=basis[I].value(self.points[J])

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

    def build_element_stiffness_matrix(self, elementID):
        if self.problemType==1:
            self.weighted_projection(elementID)
        elif self.problemType==2:
            self.scalar_elliptic(elementID)

    def build_global_stiffness_matrix(self):

        k=0
        for I in itertools.product(*[range(n) for n in self.grid.numElements]):
            S=I[::-1]
            self.build_element_stiffness_matrix(S)
            l=max(self.grid.Elements[S].stiff_data.shape)
            self.grid.global_stiff_data[k:(k+l)]=self.grid.Elements[S].stiff_data
            self.grid.global_stiff_rows[k:(k+l)]=self.grid.Elements[S].stiff_rows
            self.grid.global_stiff_cols[k:(k+l)]=self.grid.Elements[S].stiff_cols
            k=k+l

        self.grid.global_stiff_matrix=(scsp.coo_matrix((self.grid.global_stiff_data,
             (self.grid.global_stiff_rows, self.grid.global_stiff_cols)), shape=
             [np.prod(self.grid.numNodes), np.prod(self.grid.numNodes)]))

        self.grid.global_stiff_matrix=scsp.csr_matrix(self.grid.global_stiff_matrix)

    def weighted_projection(self, elementID):
        element=self.grid.Elements[elementID]
        detJ=np.linalg.det(element.A)
        nNodes=np.prod(element.numNodes)
        for i in range(nNodes):
            I=element.get_local_nodeID(i)
            for j in range(i, nNodes):
                J=element.get_local_nodeID(j)
                integral=0.
                for K in itertools.product(*[range(n) for n in self.numQuadPts]):
                    globalCoord=np.dot(element.A, self.points[K])+element.b
                    M=functions.material_scalar_value(globalCoord)
                    integral=integral+M*self.basisAtQuadPts[tuple(I)][K]*self.basisAtQuadPts[tuple(J)][K]*self.weights[K]*detJ
                element.stiff_matrix[i, j]=integral
                element.stiff_matrix[j, i]=integral

        # element.stiff_data = np.ones((np.prod(element.numNodes))**2)
        element.stiff_data=np.reshape(element.stiff_matrix, (np.prod(element.numNodes))**2)
        element.stiff_cols=np.tile(element.globalNodeNum, nNodes)
        element.stiff_rows=np.repeat(element.globalNodeNum, nNodes)

        self.grid.Elements[elementID]=element

    def scalar_elliptic(self, elementID):
        element=self.grid.Elements[elementID]
        detJ=np.linalg.det(element.A)
        nNodes=np.prod(element.numNodes)
        for i in range(nNodes):
            I=tuple(element.get_local_nodeID(i))
            for j in range(i, nNodes):
                J=tuple(element.get_local_nodeID(j))
                integral=0.0
                for K in itertools.product(*[range(n) for n in self.numQuadPts]):
                    globalCoord=np.dot(element.A, self.points[K])+element.b
                    M=functions.material_matrix_value(globalCoord)
                    integral=integral+(np.dot(np.dot(M,
                        np.dot(np.transpose(element.A_inv), self.basisGradAtQuadPts[I][K])),
                        np.dot(np.transpose(element.A_inv), self.basisGradAtQuadPts[J][K]))*
                        self.weights[K]*detJ)
                element.stiff_matrix[i, j]=integral
                element.stiff_matrix[j, i]=integral

        element.stiff_data=np.reshape(element.stiff_matrix, (np.prod(element.numNodes))**2)
        element.stiff_cols=np.tile(element.globalNodeNum, nNodes)
        element.stiff_rows=np.repeat(element.globalNodeNum, nNodes)
        self.grid.Elements[elementID]=element

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
        return self.grid.global_stiff_matrix.dot(v)

    def matrix_vector_product_krylov(self, v):

#         v_out=np.zeros(self.grid.solution_array.shape)
        v[self.homoDirichletNodeList]=0.
        v[self.inHomoDirichletNodeList]=0.
        # v_out[self.unKnownDOF] = v
        # print((self.grid.global_stiff_matrix.dot(v_out))[self.unKnownDOF].shape)
        v_out=self.grid.global_stiff_matrix.dot(v)
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

    def get_dummy_element_stiffness_matrix(self, elementID):
        element=self.grid.Elements[elementID]
        nNodes=np.prod(element.numNodes)

        element.stiff_data=np.ones((np.prod(element.numNodes))**2)
        element.stiff_cols=np.tile(element.globalNodeNum, nNodes)
        element.stiff_rows=np.repeat(element.globalNodeNum, nNodes)

        self.grid.Elements[elementID]=element

    def get_dummy_global_stiff_matrix(self):
        k=0
        for I in itertools.product(*[range(n) for n in self.grid.numElements]):
            S=I[::-1]
            self.get_dummy_element_stiffness_matrix(S)
            l=max(self.grid.Elements[S].stiff_data.shape)
            self.grid.global_stiff_data[k:(k+l)]=self.grid.Elements[S].stiff_data
            self.grid.global_stiff_rows[k:(k+l)]=self.grid.Elements[S].stiff_rows
            self.grid.global_stiff_cols[k:(k+l)]=self.grid.Elements[S].stiff_cols
            k=k+l

        self.grid.global_stiff_matrix=(scsp.coo_matrix((self.grid.global_stiff_data,
             (self.grid.global_stiff_rows, self.grid.global_stiff_cols)), shape=
             [np.prod(self.grid.numNodes), np.prod(self.grid.numNodes)]))

        self.grid.global_stiff_matrix=scsp.csr_matrix(self.grid.global_stiff_matrix)

        return self.grid.global_stiff_matrix
