# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:55:14 2018

@author: y0090873
"""

import numpy as np
from scipy.interpolate import lagrange
import itertools
# from polyNd import polyNd
# import quad as quad
import copy
import scipy.sparse as scsp


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
        self.Coord=np.zeros(dim)

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
            self.Coord[tuple(index)]=np.tile(vec, tile_size)
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
        self.Coord_double_grid=np.zeros(dim)
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
            self.Coord_double_grid[tuple(index)]=np.tile(vec, tile_size)
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
            local_node_num=local_node_num % prod
        nodeID[0]=local_node_num % self.numNodes[0]

        return nodeID

    def get_local_nodeID_double_grid(self, local_node_num):
        nodeID=np.zeros(self.dimension, dtype='int32')
        nodeID[0]=local_node_num % self.numNodes_double_grid[0]
        for i in range(self.dimension, 1,-1):
            prod=np.prod(self.numNodes_double_grid[0:(i-1)])
            nodeID[i-1]=local_node_num/prod
            local_node_num=local_node_num % prod
        nodeID[0]=local_node_num % self.numNodes_double_grid[0]

        return nodeID


class RectElement(object):

    def __init__(self, dimension, order):
        self.dimension=dimension
        assert(dimension==len(order))
        self.order=order
        if not isinstance(self.order, np.ndarray):
            self.order=np.asarray(self.order)
        self.numNodes=self.order+1
        dim=np.hstack((self.numNodes, self.dimension))
        self.Coord=np.zeros(dim)
        self.ID=np.zeros(self.dimension)
        self.globalNodeNum=np.zeros(self.order+1)
        self.stiff_data=np.zeros((np.prod(self.numNodes))**2)
        self.stiff_rows=np.zeros((np.prod(self.numNodes))**2)
        self.stiff_cols=np.zeros((np.prod(self.numNodes))**2)
        self.stiff_matrix=np.zeros((np.prod(self.numNodes), np.prod(self.numNodes)))
        self.source_vector=np.zeros(np.prod(self.order+1))
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
        dim=np.asarray(self.Coord.shape)
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
            self.Coord[tuple(index)]=np.tile(vec, tile_size)
            tile_size[i]=dim[i]

    def set_affine_mapping(self):
        self.A=np.zeros((self.dimension, self.dimension))
        self.A_inv=np.zeros((self.dimension, self.dimension))

        self.b=np.zeros(self.dimension)
        for i in range(self.dimension):
            self.A[i, i]=self.length[i]/2.0
            self.A_inv[i, i]=2.0/self.length[i]

        # self.A_inv = self.A_inv / np.linalg.det(self.A)
        self.b=self.center

    def get_local_nodeID(self, local_node_num):
        nodeID=np.zeros(self.dimension, dtype='int32')
        nodeID[0]=local_node_num % self.numNodes[0]
        for i in range(self.dimension, 1,-1):
            prod=np.prod(self.numNodes[0:(i-1)])
            nodeID[i-1]=local_node_num/prod
            local_node_num=local_node_num % prod
        nodeID[0]=local_node_num % self.numNodes[0]

        return nodeID


class boundary(object):

    def __init__(self, minCoord, maxCoord):
        assert(len(minCoord)==len(maxCoord))
        self.minCoord=np.asarray(minCoord)
        self.maxCoord=np.asarray(maxCoord)
        self.dimension=len(minCoord)-1
        check=(self.minCoord==self.maxCoord)
        assert(np.sum(check)==1)
        self.fixedDim=np.asarray(np.where(check)[0])+1


class periodic_boundaries(object):

    def __init__(self, boundary1, boundary2):
        self.boundary1=boundary1
        self.boundary2=boundary2
        assert(boundary1.dimension==boundary2.dimension)
        minCheck=(self.boundary1.minCoord!=self.boundary2.minCoord)
        maxCheck=(self.boundary1.maxCoord!=self.boundary2.maxCoord)
        assert(np.sum(minCheck)==1)
        assert(np.sum(maxCheck)==1)
        assert(self.boundary1.fixedDim==self.boundary2.fixedDim)
        self.fixedDim=self.boundary1.fixedDim


class Discretization(object):

    def __init__(self, domain):
        self.domain=domain

    def set_num_elements(self, N):
        assert(self.domain.dimension==len(N))
        self.numEle=N
        if not isinstance(self.numEle, np.ndarray):
            self.numEle=np.asarray(self.numEle)

        self.Elements=np.empty(self.numEle, RectElement)

    def set_order(self, p):
        assert(self.domain.dimension==len(p))
        self.order=p
        if not isinstance(self.order, np.ndarray):
            self.order=np.asarray(self.order)

    def compute_num_nodes(self):
        self.numNodes=np.zeros((self.domain.dimension), 'int32')
        self.numNodes=np.multiply(self.numEle, self.order)+1
        self.global_stiff_data=np.zeros((np.prod(self.numEle))*(np.prod(self.order+1))**2)
        self.global_stiff_rows=np.zeros((np.prod(self.numEle))*(np.prod(self.order+1))**2)
        self.global_stiff_cols=np.zeros((np.prod(self.numEle))*(np.prod(self.order+1))**2)
        self.global_source_vector=np.zeros(np.prod(self.numNodes))
        self.solution_array=np.zeros(np.prod(self.numNodes))
        self.B=[]

    def generate_mesh(self):
        dim=np.hstack((self.numNodes, self.domain.dimension))
        self.Coord=np.zeros(dim)
        origin=self.domain.origin
        length=self.domain.length
        self.spacing=np.divide((np.divide(length, self.numEle)), self.order)

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
            self.Coord[tuple(index)]=np.tile(vec, tile_size)
            tile_size[i]=dim[i]

    def get_element_nodes_coord(self, I):
        assert(self.domain.dimension==len(I))
        a=[None]*(len(I)+1)
        for i in range(len(I)):
            start=I[i]*self.order[i]
            stop=start+self.order[i]+1
            a[i]=slice(start, stop, 1)
        a[-1]=slice(0, len(I), 1)
        return self.Coord[tuple(a)]

    def buildElements(self):
        for I in itertools.product(*[range(n) for n in self.numEle]):
            self.Elements[I]=RectElement(self.domain.dimension, self.order)
            self.Elements[I].ID=np.asarray(I)
            self.Elements[I].globalNodeNum=(
                Discretization.get_local_to_global(self.numEle, self.Elements[I].order, I))
            Coord=self.get_element_nodes_coord(I)
            minCoord=Coord[(0,)*(self.domain.dimension)]
            maxCoord=Coord[(-1,)*(self.domain.dimension)]
            length=maxCoord-minCoord
            self.Elements[I].set_element_location(minCoord, length)
            self.Elements[I].set_affine_mapping()

    def get_global_node_num(self, nodeID):
        node_num=0
        for i in range(len(nodeID)):
            temp=nodeID[i]
            for j in range(i):
                temp=temp*self.numNodes[j]
            node_num=node_num+temp

        return int(node_num)

    def get_global_node_ID(self, node_num):
        nodeID=np.zeros(self.dimension, dtype='int32')
        nodeID[0]=node_num % self.numNodes[0]
        for i in range(self.dimension, 1,-1):
            prod=np.prod(self.numNodes[0:(i-1)])
            nodeID[i-1]=node_num/prod
            node_num=node_num%prod
        nodeID[0]=node_num % self.numNodes[0]

        return nodeID

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

    def get_elements_on_boundary(self, boundary):
        assert(self.domain.dimension==(boundary.dimension+1))
        length=np.divide(self.domain.length, self.numEle)
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
                    start[i]=self.numEle[i]-1
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

    def impose_dirichlet_BC(self, boundary, value):
        nodelist=self.get_nodes_on_boundary(boundary)
        nodelist=np.reshape(nodelist, np.prod(nodelist.shape))
        self.global_stiff_matrix=scsp.lil_matrix(self.global_stiff_matrix)

        for i in range(len(nodelist)):
            nodeNum=self.get_global_node_num(nodelist[i])
            shape=max(self.global_source_vector.shape)
            temp=np.reshape(self.global_stiff_matrix[:, nodeNum].toarray(), shape)
            self.global_source_vector=self.global_source_vector-value*temp
            self.global_stiff_matrix[:, nodeNum]=0
            self.global_stiff_matrix[nodeNum, :]=0
            self.global_stiff_matrix[nodeNum, nodeNum]=1.
            self.global_source_vector[nodeNum]=value

    def impose_periodic_BC(self, periodic_boundary):
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
            nodes2[b]=nodes1[a]
            self.Elements[id2].globalNodeNum=np.reshape(nodes2, np.prod(self.Elements[id2].numNodes), order='F')

    def solve_linear_system(self):
        self.global_stiff_matrix=scsp.csr_matrix(self.global_stiff_matrix)
        I=self.global_stiff_matrix.getnnz(1)>0
        J=self.global_stiff_matrix.getnnz(0)>0
        self.global_stiff_matrix=self.global_stiff_matrix[I][:, J]

        self.global_source_vector=self.global_source_vector[I]
        self.solution_array=self.solution_array[I]
        self.solution_array=scsp.linalg.bicgstab(self.global_stiff_matrix, self.global_source_vector)


class DoGIP(object):

    def __init__(self, grid, refele):
        self.grid=grid
        self.ref_element=refele

    def define_problem(self, problemID):
        self.problemID=problemID

    def build_A_DoGIP(self, numQuadPts):
        if self.problemID==1:
            self.ref_element.set_order_double_grid((self.ref_element.order)*2)
        if self.problemID==2:
            self.ref_element.set_order_double_grid((self.ref_element.order)*2)

        self.ref_element.set_nodal_coord_double_grid()
        self.ref_element.set_nodal_basis_double_grid()

        if not isinstance(numQuadPts, np.ndarray):
            numQuadPts=np.asarray(numQuadPts)

        [self.points, self.weights]=quad.Quadrature.get_gauss_quad(numQuadPts)
        self.basis_double_grid_at_quadpts=np.empty(self.ref_element.numNodes_double_grid, np)

        for I in itertools.product(*[range(n) for n in (self.ref_element.numNodes_double_grid)]):
            self.basis_double_grid_at_quadpts[I]=np.zeros(numQuadPts)

        basis_double_grid=self.ref_element.nodal_basis_double_grid

        for I in itertools.product(*[range(n) for n in (self.ref_element.numNodes_double_grid)]):
            for J in itertools.product(*[range(n) for n in (numQuadPts)]):
                self.basis_double_grid_at_quadpts[I][J]=basis_double_grid[I].value(self.points[J])

        for I in itertools.product(*[range(n) for n in self.grid.numEle]):
            self.build_A_T_DoGIP(I)

        self.grid.B=DoGIP.compute_interpolation_matrix(self.ref_element, self.problemID)
        self.countBC=0

    def impose_dirichlet_BC(self, boundary, value):

        nodelist=self.grid.get_nodes_on_boundary(boundary)
        nodelist=np.reshape(nodelist, np.prod(nodelist.shape))

        nodeNum=np.zeros(nodelist.shape, dtype='int32')

        for i in range(len(nodelist)):
            nodeNum[i]=self.grid.get_global_node_num(nodelist[i])

        if self.countBC==0:
            self.nodelist=nodeNum
            self.value=value
            self.count=[len(nodeNum)]
        else:
            self.nodelist=np.hstack((self.nodelist, nodeNum))
            self.value=np.hstack((self.value, value))
            self.count.append(len(nodeNum))

        self.grid.global_source_vector[self.nodelist]=value
        self.countBC+=1

    def build_A_T_DoGIP(self, elementID):
        n=np.prod(self.ref_element.numNodes_double_grid)
        a_diag=np.zeros(n)
        numQuadPts=self.weights.shape
        element=self.grid.Elements[elementID]
        detJ=np.linalg.det(element.A)

        for i in range(n):
            I=self.ref_element.get_local_nodeID_double_grid(i)
            integral=0.
            for K in itertools.product(*[range(n) for n in numQuadPts]):
                globalCoord=np.dot(element.A, self.points[K])+element.b
                M=stiffness_matrix.material_scalar_value(globalCoord)
                integral=integral+M*self.basis_double_grid_at_quadpts[tuple(I)][K]*self.weights[K]*detJ
            a_diag[i]=integral

        element.A_T_DoGIP=scsp.spdiags(a_diag, 0, n, n)

        self.grid.Elements[elementID]=element

    @staticmethod
    def compute_interpolation_matrix(ref_element, problemID):

        nodes_double_grid=np.prod(ref_element.numNodes_double_grid)
        nodes=np.prod(ref_element.numNodes)
        basis=ref_element.nodal_basis

        if problemID==1:
            B_hat=scsp.lil_matrix((nodes_double_grid, nodes))

        if problemID==1:
            i=0
            for I in itertools.product(*[range(n) for n in ref_element.numNodes]):
                S=tuple(I[::-1])
                j=0
                for J in itertools.product(*[range(n) for n in ref_element.numNodes_double_grid]):
                    T=tuple(J[::-1])
                    B_hat[j, i]=basis[S].value(ref_element.Coord_double_grid[T])
                    j+=1
                i+=1

        B_hat=B_hat.tocsr()
        return B_hat

    def matrix_vector_mult_krylov(self, v):
        v_temp=np.zeros(v.shape)
        for I in itertools.product(*[range(n) for n in self.grid.numEle]):
            u=v[self.grid.Elements[I].globalNodeNum]
            temp=np.dot(self.grid.B, u)
            temp=self.grid.Elements[I].A_T_DoGIP.dot(temp)
            temp=np.dot(self.grid.B.T, temp)
            for i in range(len(temp)):
                v_temp[self.grid.Elements[I].globalNodeNum[i]]+=temp[i]

        k=0
        for i in range(len(self.count)):
            v_temp[self.nodelist[k:k+self.count[i]]]=self.value[i]
            k=k+self.count[i]

        return v_temp

    def solve_linear_system(self):
        self.A=scsp.linalg.LinearOperator(shape=(np.prod(self.grid.numNodes), np.prod(self.grid.numNodes)), matvec=self.matrix_vector_mult_krylov)
        self.grid.solution_array=scsp.linalg.gmres(self.A, self.grid.global_source_vector)


class stiffness_matrix(object):

    def __init__(self, grid, refele):
        self.grid=grid
        self.ref_element=refele

    def build_element_stiff_matrix(self, functionID, numQuadPts, elementID):
        dimension=self.grid.domain.dimension
        assert(len(numQuadPts)==dimension)

        if functionID==1:
            self.bilinear_weighted_proj(elementID)
        if functionID==2:
            self.bilinear_poisson(elementID)

    @staticmethod
    def material_matrix_value(Coord):
        dim=len(Coord)
        value=np.eye(dim)
        return value

    @staticmethod
    def material_scalar_value(Coord):
        value=1e7
        return value

    def bilinear_weighted_proj(self, elementID):
        # numQuadPoints = self.weights.shape
        element=self.grid.Elements[elementID]
        # detJ = np.linalg.det(element.A)
        nNodes=np.prod(element.numNodes)
#        for i in range(nNodes):
#            I = element.get_local_nodeID(i)
#            for j in range(i,nNodes):
#                J = element.get_local_nodeID(j)
#                integral = 0.
#                for K in itertools.product(*[range(n) for n in numQuadPoints]):
#                    globalCoord = np.dot(element.A, self.points[K]) + element.b
#                    M = stiffness_matrix.material_scalar_value(globalCoord)
#                    integral = integral + M * self.basis_at_quadpts[tuple(I)][K] * self.basis_at_quadpts[tuple(J)][K] * self.weights[K] * detJ
#                element.stiff_matrix[i,j] = integral
#                element.stiff_matrix[j,i] = integral

        element.stiff_data=np.ones((np.prod(element.numNodes))**2)
        # element.stiff_data = np.reshape(element.stiff_matrix, (np.prod(element.numNodes))**2)
        element.stiff_cols=np.tile(element.globalNodeNum, nNodes)
        element.stiff_rows=np.repeat(element.globalNodeNum, nNodes)
        self.grid.Elements[elementID]=element

    def bilinear_poisson(self, elementID):
        numQuadPoints=self.weights.shape
        element=self.grid.Elements[elementID]
        detJ=np.linalg.det(element.A)
        nNodes=np.prod(element.numNodes)
        for i in range(nNodes):
            I=tuple(element.get_local_nodeID(i))
            for j in range(i, nNodes):
                J=tuple(element.get_local_nodeID(j))
                integral=0.0
                for K in itertools.product(*[range(n) for n in numQuadPoints]):
                    globalCoord=np.dot(element.A, self.points[K])+element.b
                    M=stiffness_matrix.material_matrix_value(globalCoord)
                    integral=integral+(np.dot(np.dot(M,
                        np.dot(np.transpose(element.A_inv), self.basis_grad_at_quadpts[I][K])),
                        np.dot(np.transpose(element.A_inv), self.basis_grad_at_quadpts[J][K]))*
                        self.weights[K]*detJ)
                element.stiff_matrix[i, j]=integral
                element.stiff_matrix[j, i]=integral

        element.stiff_data=np.reshape(element.stiff_matrix, (np.prod(element.numNodes))**2)
        element.stiff_cols=np.tile(element.globalNodeNum, nNodes)
        element.stiff_rows=np.repeat(element.globalNodeNum, nNodes)
        self.grid.Elements[elementID]=element

    def build_global_stiff_matrix(self, functionID, numQuadPts):

        if not isinstance(numQuadPts, np.ndarray):
            numQuadPts=np.asarray(numQuadPts)

        [self.points, self.weights]=quad.Quadrature.get_gauss_quad(numQuadPts)
        self.basis_at_quadpts=np.empty(self.grid.order+1, np)
        # self.basis_grad_at_quadpts = np.empty(self.grid.order + 1, np)

        for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
            self.basis_at_quadpts[I]=np.zeros(numQuadPts)
            # self.basis_grad_at_quadpts[I] = np.zeros(np.hstack((numQuadPts, self.grid.domain.dimension)))

        basis=self.ref_element.nodal_basis
        # basis_grad = self.ref_element.nodal_basis_gradient

        for I in itertools.product(*[range(n) for n in (self.grid.order+1)]):
            for J in itertools.product(*[range(n) for n in (numQuadPts)]):
                self.basis_at_quadpts[I][J]=basis[I].value(self.points[J])
                # self.basis_grad_at_quadpts[I][J] = np.asarray(polyNd.get_value_for_polyNdarray(basis_grad[I], self.points[J]))

        k=0
        for I in itertools.product(*[range(n) for n in self.grid.numEle]):
            S=I[::-1]
            self.build_element_stiff_matrix(functionID, numQuadPts, S)
            l=max(self.grid.Elements[S].stiff_data.shape)
            self.grid.global_stiff_data[k:(k+l)]=self.grid.Elements[S].stiff_data
            self.grid.global_stiff_rows[k:(k+l)]=self.grid.Elements[S].stiff_rows
            self.grid.global_stiff_cols[k:(k+l)]=self.grid.Elements[S].stiff_cols
            k=k+l

        self.grid.global_stiff_matrix=(scsp.coo_matrix((self.grid.global_stiff_data,
             (self.grid.global_stiff_rows, self.grid.global_stiff_cols)), shape=
             [np.prod(self.grid.numNodes), np.prod(self.grid.numNodes)]))

        self.grid.global_stiff_matrix=scsp.csr_matrix(self.grid.global_stiff_matrix)


class source_vector(object):

    def __init__(self, grid, refele):
        self.grid=grid
        self.ref_element=refele
        self.firstCall=True

    def build_element_source_vector(self, function, numQuadPts, elementID):
        dimension=self.grid.domain.dimension
        assert(len(numQuadPts)==dimension)
        if self.firstCall is True:
            [self.points, self.weights]=quad.Quadrature.get_gauss_quad(numQuadPts)
            self.firstCall=False

        function(self, elementID)

    @staticmethod
    def source_func_value(Coord):
        value=1e7
        return value

    def linear_function(self, elementID):
        numQuadPoints=self.weights.shape
        element=self.grid.Elements[elementID]
        detJ=np.linalg.det(element.A)
        basis=self.ref_element.nodal_basis
        nNodes=np.prod(element.numNodes)
        for i in range(nNodes):
            I=element.get_local_nodeID(i)
            basis1=basis[tuple(I)]
            integral=0.
            for K in itertools.product(*[range(n) for n in numQuadPoints]):
                globalCoord=np.dot(element.A, self.points[K])+element.b
                f=source_vector.source_func_value(globalCoord)
                integral=integral+f*basis1.value(self.points[K])*self.weights[K]*detJ
            element.source_vector[i]=integral

        self.grid.Elements[elementID]=element

    def build_global_source_vector(self, numQuadPts):

        for I in itertools.product(*[range(n) for n in self.grid.numEle]):
            globalNodes=self.grid.Elements[I].globalNodeNum
            self.build_element_source_vector(source_vector.linear_function, numQuadPts, I)
            for i in range(len(globalNodes)):
                self.grid.global_source_vector[globalNodes[i]]+=self.grid.Elements[I].source_vector[i]
