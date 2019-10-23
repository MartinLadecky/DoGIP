# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:36:37 2019

@author: rajuc
"""

import numpy as np
from scipy.interpolate import lagrange
import itertools
from PreProcessing.polyNd import polyNd
import copy


class Domain(object):

    def __init__(self, dimension):
        self.dimension=dimension

    def set_origin(self, origin):
        assert(len(origin)==self.dimension)
        self.origin=origin
        if not isinstance(self.origin, np.ndarray):
            self.origin=np.asarray(self.origin)

    def set_domain_length(self, length):
        assert(len(length)==self.dimension)
        self.length=length
        if not isinstance(self.length, np.ndarray):
            self.length=np.asarray(self.length)


class RectReferenceElement(object):

    def __init__(self, dimension, order):
        self.dimension=dimension
        assert(self.dimension==len(order))
        self.order=order
        if not isinstance(self.order, np.ndarray):
            self.order=np.asarray(self.order)
        self.numNodes=np.zeros(self.dimension, 'int32')
        for i in range(self.dimension):
            self.numNodes[i]=self.order[i]+1

    def set_order(self, order):
        assert(self.dimension==len(order))
        self.order=order
        if not isinstance(self.order, np.ndarray):
            self.order=np.asarray(self.order)
        self.numNodes=self.order+1

    def set_order_double_grid(self, order):
        assert(self.dimension==len(order))
        self.order_double_grid=order
        if not isinstance(self.order_double_grid, np.ndarray):
            self.order_double_grid=np.asarray(self.order_double_grid)
        self.numNodes_double_grid=self.order_double_grid+1

    def set_nodal_coord(self):
        dim=np.hstack((self.numNodes, self.dimension))

        # 4D array of nodal coordinates. size:(nx,ny,nz,3)
        self.coord=np.zeros(dim)

        # slicing index
        index=[None]*(self.dimension+1)

        tile_size=copy.deepcopy(dim)
        for i in range(self.dimension):
            vec=-1.0+(2.0/self.order[i])*np.arange(0, self.numNodes[i])
            vec_size=np.ones(self.dimension+1, dtype='int32')
            vec_size[i]=self.numNodes[i]
            vec=np.reshape(vec, vec_size)
            tile_size[i]=1
            tile_size[-1]=1
            for j in range(self.dimension):
                index[j]=slice(0, self.numNodes[j], 1)
            index[-1]=slice(i, i+1, 1)
            self.coord[tuple(index)]=np.tile(vec, tile_size)
            tile_size[i]=dim[i]

    def set_nodal_basis(self):
        poly1darraylist=np.empty(self.dimension, np)
        for i in range(self.dimension):
            poly1darraylist[i]=np.empty(self.numNodes[i], np)
        for i in range(self.dimension):
            coord=np.arange(-1., 1.+2./self.order[i], 2./self.order[i])
            val=np.zeros(self.numNodes[i])
            for j in range(self.numNodes[i]):
                val[j]=1.
                (poly1darraylist[i])[j]=lagrange(coord, val)
                val[j]=0
        self.nodal_basis=polyNd.tensorProductPoly1d(poly1darraylist)

    def set_nodal_coord_double_grid(self):
        dim=np.hstack((self.numNodes_double_grid, self.dimension))
        self.coord_double_grid=np.zeros(dim)
        index=[None]*(self.dimension+1)
        tile_size=copy.deepcopy(dim)
        for i in range(self.dimension):
            vec=-1.0+(2.0/self.order_double_grid[i])*np.arange(0, self.numNodes_double_grid[i])
            vec_size=np.ones(self.dimension+1, dtype='int32')
            vec_size[i]=self.numNodes_double_grid[i]
            vec=np.reshape(vec, vec_size)
            tile_size[i]=1
            tile_size[-1]=1
            for j in range(self.dimension):
                index[j]=slice(0, self.numNodes_double_grid[j], 1)
            index[-1]=slice(i, i+1, 1)
            self.coord_double_grid[tuple(index)]=np.tile(vec, tile_size)
            tile_size[i]=dim[i]

    def set_nodal_basis_double_grid(self):
        poly1darraylist=np.empty(self.dimension, np)
        for i in range(self.dimension):
            poly1darraylist[i]=np.empty(self.numNodes_double_grid[i], np)
        for i in range(self.dimension):
            coord=np.arange(-1., 1.+2./self.order_double_grid[i], 2./self.order_double_grid[i])
            val=np.zeros(self.numNodes_double_grid[i])
            for j in range(self.numNodes_double_grid[i]):
                val[j]=1.
                (poly1darraylist[i])[j]=lagrange(coord, val)
                val[j]=0
        self.nodal_basis_double_grid=polyNd.tensorProductPoly1d(poly1darraylist)

    def set_first_order_derivatives(self):
        self.nodal_basis_gradient=np.empty(self.order+1, np)

        for I in itertools.product(*[range(n) for n in self.order+1]):
            self.nodal_basis_gradient[I]=polyNd.partialderivative(self.nodal_basis[I])

    def get_local_nodeID(self, local_node_num):
        nodeID=np.zeros(self.dimension, dtype='int32')
        nodeID[0]=local_node_num % self.numNodes[0]
        for i in range(self.dimension, 1,-1):
            prod=np.prod(self.numNodes[0:(i-1)])
            nodeID[i-1]=local_node_num/prod
            local_node_num=local_node_num%prod
        nodeID[0]=local_node_num % self.numNodes[0]

        return nodeID

    def get_local_nodeID_double_grid(self, local_node_num):
        nodeID=np.zeros(self.dimension, dtype='int32')
        nodeID[0]=local_node_num % self.numNodes_double_grid[0]
        for i in range(self.dimension, 1,-1):
            prod=np.prod(self.numNodes_double_grid[0:(i-1)])
            nodeID[i-1]=local_node_num/prod
            local_node_num=local_node_num%prod
        nodeID[0]=local_node_num % self.numNodes_double_grid[0]

        return nodeID

    def initialize_interpolation_arrays(self, solution_approach=2, problem_type=1):

        self.B=[]
        self.B_adj=[]

        if solution_approach==3:
            self.B_1=[]
            self.B_2=[]
#        if problem_type == 1:
#            if solution_approach == 2:
#                self.B = np.zeros((np.prod(self.numNodes_double_grid), np.prod(self.numNodes)))
#            elif solution_approach == 3:
#                self.B = np.zeros((self.numNodes_double_grid[0], self.numNodes[0]))
#
#        elif problem_type == 2:
#            if solution_approach == 2:
#                self.B = []
#                for i in range(self.dimension):
#                    self.B.append(np.zeros((np.prod(self.numNodes_double_grid), np.prod(self.numNodes))))
#
#            elif solution_approach == 3:
#                self.B_1 = np.zeros((self.numNodes_double_grid[0], self.numNodes[0]))
#                self.B_2 = np.zeros((self.numNodes_double_grid[0], self.numNodes[0]))


class RectElement(object):

    def __init__(self, dimension, order):
        self.dimension=dimension
        assert(dimension==len(order))
        self.order=order
        if not isinstance(self.order, np.ndarray):
            self.order=np.asarray(self.order)
        self.numNodes=self.order+1
        dim=np.hstack((self.numNodes, self.dimension))
        self.coord=np.zeros(dim)
        self.globalNodeNum=np.zeros(np.prod(self.order+1))
        self.globalNodeID=[]
        self.globalElementID=np.zeros(self.dimension)
        self.globalElementNum=0

    def initialize_arrays(self, solution_approach=1, problem_type=1):

        self.source_vector=np.zeros(np.prod(self.order+1))

        if solution_approach==1:
            self.stiff_data=np.zeros((np.prod(self.numNodes))**2)
            self.stiff_rows=np.zeros((np.prod(self.numNodes))**2)
            self.stiff_cols=np.zeros((np.prod(self.numNodes))**2)
            self.stiff_matrix=np.zeros((np.prod(self.numNodes), np.prod(self.numNodes)))

        elif solution_approach==2 or solution_approach==3:
            self.A_T_DoGIP=[]

    def set_element_location(self, minCoord, length):

        self.length=length
        self.minCoord=minCoord

        if not isinstance(self.length, np.ndarray):
            self.length=np.asarray(self.length)
        if not isinstance(self.minCoord, np.ndarray):
            self.minCoord=np.asarray(self.minCoord)

        assert(max(self.minCoord.shape)==self.dimension)
        assert(max(self.length.shape)==self.dimension)
        self.maxCoord=self.minCoord+self.length
        self.center=(self.minCoord+self.maxCoord)/2.0

        self.spacing=np.divide(length, self.order)
        dim=np.asarray(self.coord.shape)
        index=[None]*(self.dimension+1)
        tile_size=copy.deepcopy(dim)
        for i in range(self.dimension):
            vec=self.minCoord[i]+self.spacing[i]*np.arange(0, self.numNodes[i])
            vec_size=np.ones(self.dimension+1, dtype='int32')
            vec_size[i]=self.numNodes[i]
            vec=np.reshape(vec, vec_size)
            tile_size[i]=1
            tile_size[-1]=1
            for j in range(self.dimension):
                index[j]=slice(0, self.numNodes[j], 1)
            index[-1]=slice(i, i+1, 1)
            self.coord[tuple(index)]=np.tile(vec, tile_size)
            tile_size[i]=dim[i]

    def set_affine_mapping(self):
        self.A=np.zeros((self.dimension, self.dimension))
        self.A_inv=np.zeros((self.dimension, self.dimension))

        self.b=np.zeros(self.dimension)
        for i in range(self.dimension):
            self.A[i, i]=self.length[i]/2.0
            self.A_inv[i, i]=2.0/self.length[i]

        # self.A_inv = self.A_inv / np.linalg.det(self.A)
        self.b=copy.deepcopy(self.center)

    def get_local_nodeID(self, local_node_num):
        node_ID=np.zeros(self.dimension, dtype='int32')
        node_ID[0]=local_node_num % self.numNodes[0]
        for i in range(self.dimension, 1,-1):
            prod=np.prod(self.numNodes[0:(i-1)])
            node_ID[i-1]=local_node_num/prod
            local_node_num=local_node_num%prod
        node_ID[0]=local_node_num % self.numNodes[0]

        return node_ID

    def get_local_node_num(self, local_node_ID):
        node_num=0
        for i in range(len(local_node_ID)):
            temp=local_node_ID[i]
            for j in range(i):
                temp=temp*self.numNodes[j]
            node_num=node_num+temp

        return int(node_num)


class Discretization(object):

    def __init__(self, domain):
        self.domain=domain

    def set_num_elements(self, N):
        assert(self.domain.dimension==len(N))
        self.numElements=N
        if not isinstance(self.numElements, np.ndarray):
            self.numElements=np.asarray(self.numElements)

        self.Elements=np.empty(self.numElements, RectElement)

    def set_order(self, p):
        assert(self.domain.dimension==len(p))
        self.order=p
        if not isinstance(self.order, np.ndarray):
            self.order=np.asarray(self.order)

    def generate_mesh(self):

        self.numNodes=np.zeros((self.domain.dimension), 'int32')
        self.numNodes=np.multiply(self.numElements, self.order)+1

        dim=np.hstack((self.numNodes, self.domain.dimension))
        self.coord=np.zeros(dim)
        origin=self.domain.origin
        length=self.domain.length
        self.spacing=np.divide((np.divide(length, self.numElements)), self.order)

        index=[None]*(self.domain.dimension+1)
        tile_size=copy.deepcopy(dim)
        for i in range(self.domain.dimension):
            vec=origin[i]+self.spacing[i]*np.arange(0, self.numNodes[i])
            vec_size=np.ones(self.domain.dimension+1, dtype='int32')
            vec_size[i]=self.numNodes[i]
            vec=np.reshape(vec, vec_size)
            tile_size[i]=1
            tile_size[-1]=1
            for j in range(self.domain.dimension):
                index[j]=slice(0, self.numNodes[j], 1)
            index[-1]=slice(i, i+1, 1)
            self.coord[tuple(index)]=np.tile(vec, tile_size)
            tile_size[i]=dim[i]

        self.set_order_double_grid()
        self._buildElements()

        self.periodicNodes=[]

    def get_element_nodes_coord(self, I):
        assert(self.domain.dimension==len(I))
        a=[None]*(len(I)+1)
        for i in range(len(I)):
            start=I[i]*self.order[i]
            stop=start+self.order[i]+1
            a[i]=slice(start, stop, 1)
        a[-1]=slice(0, len(I), 1)
        return self.coord[tuple(a)]

    def _build_reference_element(self):
        self.referenceElement=RectReferenceElement(self.domain.dimension, self.order)
        self.referenceElement.set_nodal_coord()
        self.referenceElement.set_nodal_basis()

    def _buildElements(self):
        for I in itertools.product(*[range(n) for n in self.numElements]):
            self.Elements[I]=RectElement(self.domain.dimension, self.order)
            self.Elements[I].globalElementID=np.asarray(I)
            self.Elements[I].globalElementNum=self.get_global_element_num(I)
            self.Elements[I].globalNodeNum=(
                Discretization.get_local_to_global(self.numElements, self.Elements[I].order, I))
            for i in range(np.prod(self.order+1)):
                self.Elements[I].globalNodeID.append(self.get_global_node_ID(self.Elements[I].globalNodeNum[i]))

            coord=self.get_element_nodes_coord(I)
            minCoord=coord[(0,)*(self.domain.dimension)]
            maxCoord=coord[(-1,)*(self.domain.dimension)]
            length=maxCoord-minCoord
            self.Elements[I].set_element_location(minCoord, length)
            self.Elements[I].set_affine_mapping()

    def get_global_node_num(self, node_ID):
        node_num=0
        for i in range(len(node_ID)):
            temp=node_ID[i]
            for j in range(i):
                temp=temp*self.numNodes[j]
            node_num=node_num+temp

        return int(node_num)

    def get_global_node_ID(self, node_num):
        node_ID=np.zeros(self.domain.dimension, dtype='int32')
        node_ID[0]=node_num % self.numNodes[0]
        for i in range(self.domain.dimension, 1,-1):
            prod=np.prod(self.numNodes[0:(i-1)])
            node_ID[i-1]=node_num/prod
            node_num=node_num % prod
        node_ID[0]=node_num % self.numNodes[0]

        return node_ID

    def get_global_element_ID(self, element_num):
        element_ID=np.zeros(self.domain.dimension, dtype='int32')
        element_ID[0]=element_num % self.numElements[0]
        for i in range(self.domain.dimension, 1,-1):
            prod=np.prod(self.numElements[0:(i-1)])
            element_ID[i-1]=element_num/prod
            element_num=element_num % prod
        element_ID[0]=element_num % self.numElements[0]

        return element_ID

    def get_global_element_num(self, element_ID):
        element_num=0
        for i in range(len(element_ID)):
            temp=element_ID[i]
            for j in range(i):
                temp=temp*self.numElements[j]
            element_num=element_num+temp

        return int(element_num)

    def initialize_arrays(self, solution_approach=1, problem_type=1):

        for I in itertools.product(*[range(n) for n in self.numElements]):
            self.Elements[I].initialize_arrays(solution_approach)

        if solution_approach==2 or solution_approach==3:
            self.referenceElement.set_order_double_grid(self.order*2)
            self.referenceElement.set_nodal_coord_double_grid()
            self.referenceElement.set_nodal_basis_double_grid()
            self.referenceElement.initialize_interpolation_arrays(solution_approach, problem_type)

        if problem_type==2:
            self.referenceElement.set_first_order_derivatives()

        if solution_approach==1:
            self.global_stiff_data=np.zeros((np.prod(self.numElements))*(np.prod(self.order+1))**2)
            self.global_stiff_rows=np.zeros((np.prod(self.numElements))*(np.prod(self.order+1))**2)
            self.global_stiff_cols=np.zeros((np.prod(self.numElements))*(np.prod(self.order+1))**2)
            self.global_stiff_matrix=[]

        self.global_source_vector=np.zeros((np.prod(self.numNodes)))
        self.solution_array=np.zeros((np.prod(self.numNodes)))
        self.solution_array_g=np.zeros((np.prod(self.numNodes)))

    @staticmethod
    def get_local_to_global(N, p, I):
        assert(len(N)==len(p))
        dim=len(I)
        assert(dim==len(N))
        if not isinstance(N, np.ndarray):
            N=np.asarray(N)
        if not isinstance(p, np.ndarray):
            p=np.asarray(p)
        numNodesPerElement=np.prod(p+1)
        globalNodes=np.arange(numNodesPerElement)
        for j in range(numNodesPerElement):
            count=0
            for i in range(dim):
                prod=np.prod(np.multiply(N[0:i], p[0:i])+1)
                count=(count+prod*(I[i]*p[i]+
                int((j % (np.prod(p[0:(i+1)]+1)))/(np.prod(p[0:i]+1)))))
            globalNodes[j]=count+1
        return globalNodes-1

    def make_periodic_boundaries(self, periodic_boundary):
        boundary1=periodic_boundary.boundary1
        boundary2=periodic_boundary.boundary2
        elelist1=self.get_elements_on_boundary(boundary1)
        elelist2=self.get_elements_on_boundary(boundary2)
        fixedDim=periodic_boundary.fixedDim
        count=elelist1.shape
        a=[None]*(self.domain.dimension)
        b=[None]*(self.domain.dimension)

        for i in range(self.domain.dimension):
            if i!=(fixedDim-1):
                a[i]=slice(0, self.order[i]+1, 1)
                b[i]=slice(0, self.order[i]+1, 1)
            else:
                a[i]=slice(0, 1, 1)
                b[i]=slice(self.order[i], self.order[i]+1, 1)

        for I in itertools.product(*[range(n) for n in count]):
            id1=tuple(elelist1[I])
            id2=tuple(elelist2[I])
            nodes1=np.reshape(self.Elements[id1].globalNodeNum, self.Elements[id1].numNodes, order='F')
            nodes2=np.reshape(self.Elements[id2].globalNodeNum, self.Elements[id2].numNodes, order='F')
            rNodes=nodes2[tuple(b)]
            cNodes=nodes1[tuple(a)]

            rNodes=np.reshape(rNodes, rNodes.size, order='F')
            cNodes=np.reshape(cNodes, cNodes.size, order='F')

            for i in range(len(rNodes)):
                self.periodicNodes.append(tuple((cNodes[i], rNodes[i])))

            nodes2[tuple(b)]=nodes1[tuple(a)]
            self.Elements[id2].globalNodeNum=np.reshape(nodes2, np.prod(self.Elements[id2].numNodes), order='F')

    def get_elements_on_boundary(self, boundary):
        assert(self.domain.dimension==(boundary.dimension+1))
        length=np.divide(self.domain.length, self.numElements)
        count=np.zeros(self.domain.dimension, dtype='int32')
        start=np.zeros(self.domain.dimension, dtype='int32')

        for i in range(self.domain.dimension):
            if i!=(boundary.fixedDim-1):
                count[i]=int((boundary.maxCoord[i]-boundary.minCoord[i])/length[i])
                start[i]=int((boundary.minCoord[i]-self.domain.origin[i])/length[i])
            else:
                count[i]=1
                if boundary.minCoord[i]==self.domain.origin[i]:
                    start[i]=0
                else:
                    start[i]=self.numElements[i]-1
        elementlist=np.empty(count, np)
        for I in itertools.product(*[range(n) for n in count]):
            elementlist[I]=np.zeros(self.domain.dimension)
            elementlist[I]=start+np.asarray(I)

        return elementlist

    def get_nodes_on_boundary(self, boundary):
        assert(self.domain.dimension==(boundary.dimension+1))
        count=np.zeros(self.domain.dimension, dtype='int32')
        start=np.zeros(self.domain.dimension, dtype='int32')

        for i in range(self.domain.dimension):
            if i!=(boundary.fixedDim-1):
                count[i]=int((boundary.maxCoord[i]-boundary.minCoord[i])/self.spacing[i])+1
                start[i]=int((boundary.minCoord[i]-self.domain.origin[i])/self.spacing[i])
            else:
                count[i]=1
                if boundary.minCoord[i]==self.domain.origin[i]:
                    start[i]=0
                else:
                    start[i]=self.numNodes[i]-1
        nodelist=np.empty(count, np)
        for I in itertools.product(*[range(n) for n in count]):
            nodelist[I]=np.zeros(self.domain.dimension)
            nodelist[I]=start+np.asarray(I)

        return nodelist

#    def impose_dirichlet_BC(self, boundary, value):
#        nodelist = self.get_nodes_on_boundary(boundary)
#        nodelist = np.reshape(nodelist, np.prod(nodelist.shape))
#        self.global_stiff_matrix = scsp.lil_matrix(self.global_stiff_matrix)
#
#        for i in range(len(nodelist)):
#            nodeNum = self.get_global_node_num(nodelist[i])
#            shape = max(self.global_source_vector.shape)
#            temp = np.reshape(self.global_stiff_matrix[:,nodeNum].toarray(), shape)
#            self.global_source_vector = self.global_source_vector - value * temp
#            self.global_stiff_matrix[:,nodeNum] = 0
#            self.global_stiff_matrix[nodeNum, :] = 0
#            self.global_stiff_matrix[nodeNum, nodeNum] = 1.
#            self.global_source_vector[nodeNum] = value
#

#    def solve_linear_system(self):
#        self.global_stiff_matrix = scsp.csr_matrix(self.global_stiff_matrix)
#        I = self.global_stiff_matrix.getnnz(1) > 0
#        J = self.global_stiff_matrix.getnnz(0) > 0
#        self.global_stiff_matrix = self.global_stiff_matrix[I][:,J]
#
#        self.global_source_vector = self.global_source_vector[I]
#        self.solution_array = self.solution_array[I]
#        self.solution_array = scsp.linalg.bicgstab(self.global_stiff_matrix, self.global_source_vector)
