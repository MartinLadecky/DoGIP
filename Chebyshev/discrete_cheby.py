import numpy as np
from scipy.fftpack import dct, idct, dctn, idctn
from scipy.interpolate import lagrange
import itertools
import copy
import matplotlib.pyplot as plt
from Chebyshev.DCTs import * # dctn_, idctn_, grad_,grad_adjoin_, decrease_spectrum, enlarge_spectrum, divide_both ,\
       # dctn_ortho_, idctn_ortho_ ,dctn_trans_, idctn_trans_
def f_x(x_k):
   # squarer=lambda t: t**3+np.cos(3*t)+np.exp(-(t**7))
    squarer = lambda t: (t-1)**2+3
    f=np.vectorize(squarer)
    fx=f(x_k)
    return fx

def cheb_extrema_grid1D(N): ## points are ordered from 1 to -1
    # N is number of points
    x_k=[]
    for k in range(0,N):
        x_k.append(np.cos((k*np.pi)/(N-1)))
    return x_k

def cheb_extrema_gridnD(N):
    # N is number of points
    #x_k,y_k,z_k =np.zeros([N[0]]),np.zeros([N[1]]),np.zeros([N[2] ])

    dim = np.size(N)
    coord = np.hstack((N, dim))
    x=np.empty(N)
    index = [None]*(dim+1)
    tile_size = copy.deepcopy(coord)

    return coord



def build_DCT_extrema1D(N):
    tCh=np.zeros(shape=(N, N))
    for m in range(0,N):
        #tCh_coef=np.zeros(shape=(N))
        #tCh_coef[m]=1
        for k in range(0,N):
            tCh[m,k]=((-1)**m)*np.cos(np.pi*m*k/(N-1))
    return tCh

def build_DCT_text_1D(N):
    tCh=np.zeros(shape=(N, N))
    for m in range(0,N):
        #tCh_coef=np.zeros(shape=(N))
        #tCh_coef[m]=1

        for k in range(0,N):
            tCh[m,k]=(2/(N-1))*(np.cos(np.pi*m*k/(N-1)))
            if k==0 or k == N-1:
                tCh[m, k] = tCh[m, k]/2
    return tCh

def build_DCT_ortho_1D(N):
    tCh=np.zeros(shape=(N, N))
    for m in range(0,N):
        #tCh_coef=np.zeros(shape=(N))
        #tCh_coef[m]=1

        for k in range(0,N):
            tCh[m,k]=(np.cos(np.pi*m*k/(N-1)))*np.sqrt(2/(N-1))#*np.sqrt(2)
            if k==0 or k == N-1:
                tCh[m, k] = tCh[m, k]*np.sqrt(2)/(2)

            if m==0 or m == N-1:
                tCh[m, k] = tCh[m, k]/(np.sqrt(2))
              #  tCh[m, k] = tCh[m, k]*(np.sqrt(2))
    return tCh




def build_iDCT_text_1D(N):
    tCh=np.zeros(shape=(N, N))
    for m in range(0,N):
        #tCh_coef=np.zeros(shape=(N))
        #tCh_coef[m]=1

        for k in range(0,N):
            tCh[m,k]=(np.cos(np.pi*m*k/(N-1)))
            if k==0 or k == N-1:
                tCh[m, k] = tCh[m, k]/2
    return tCh

def compute_DCT_ortho1D(u):
    N=np.size(u)
    T=build_DCT_extrema1D(N)

    a=2*((T.dot(u))/(N-1))

    a[0]=a[0]/2
    a[-1]=a[-1]/2

    return a

def compute_DCT_extrema1D(u):
    N=np.size(u)
    T=build_DCT_extrema1D(N)

    a=2*((T.dot(u))/(N-1))

    a[0]=a[0]/2
    a[-1]=a[-1]/2

    return a

def build_DCT_extrema1D_new(N):
    tCh=np.zeros(shape=(N+1, N+1))
    for m in range(0,N+1):
        #tCh_coef=np.zeros(shape=(N))
        #tCh_coef[m]=1
        for k in range(0,N+1):
            tCh[m,k]=2*np.cos(np.pi*m*k/(N))
            if k == 0:
                tCh[m, k]=tCh[m, k]/2
            elif k == N:
                tCh[m, k]=tCh[m, k]/2
        #if m == 0:
         #   tCh[m, k]=tCh[m, k]/2
        #elif m == N:
        #    tCh[m, k]=tCh[m, k]/2



    return tCh ##/(N+1)


def compute_iDCT_extrema1D(a):
    N=np.size(a)
    T=build_DCT_extrema1D(N)
    u=(np.matrix.transpose(T).dot(a))
    u[0]=u[0]/2
    u[-1]=u[-1]/2

    return u

def cheb_root_grid1D(N):
    x_k=[]
    for k in range(0,N):
       #3 x_k.append(-np.cos(((2*k+1)*np.pi)/(2*(N))))
        x_k.append(-np.cos((np.pi/(N-1))*(k+1/2)))
    return x_k

def build_DCT_roots1D(N):
    tCh=np.zeros(shape=(N, N))
    tChq=np.zeros(shape=(N, N))
    for m in range(0,N):
        tCh_coef=np.zeros(shape=(N))
        tCh_coef[m]=1
        for k in range(0,N):
            #tCh[m,k]=chebyshev.chebval(x_k[k],tCh_coef)
            tCh[m, k]=np.cos((m*np.pi/(N-1))*((N-1)+k+1/2))

           # tCh[m, k]=np.cos(m/(np.cos(-np.cos((k+0.5)*np.pi/(N-1)))))

    return tCh

def compute_DCT_roots1D(u):
    N=np.size(u)
    T=build_DCT_roots1D(N)
    a=2*((T.dot(u))/(N-1))
    a[0]=a[0]/2
    a[-1]=a[-1]/2

    return a

def compute_iDCT_roots1D(a):
    N=np.size(a)
    T=build_DCT_roots1D(N)
    u=(np.matrix.transpose(T).dot(a))
    u[0]=u[0]/2
    u[-1]=u[-1]/2

    return u

#
def dct_cheb(u):
    a = dct(u,type=1)
    return a


def idct_chebDG(a):
    N = np.size(a)
    a[-1] = a[-1]/2
    a_ext = np.pad(a, (0, np.size(a)-1), 'constant')
    u_ext = idct(a_ext, type=1)/(2*(N-1))
    return u_ext



#
def nd_DCT(a):
    dim = np.size(np.shape(a))
    Ns = np.shape(a)

    factor=np.ones(1)
    for d in range(dim):
        factor=factor*(Ns[d]-1)

    a_hat=dctn(a,type=1)/((2**dim)*factor)

    return a_hat

def nd_iDCT(a_hat):
    N=np.size(a_hat)
    a=idctn(a_hat,type=1)
    return a


def nd_extend(a_hat):
    # extend spectrum and divide middle frequency to keep result correct
    dim =np.size( np.shape(a_hat))

    if dim ==2:
        a_hat[-1, :]=a_hat[-1, :]/2
        a_hat[:, -1]=a_hat[:, -1]/2
        a_ext=np.pad(a_hat, [(0, np.shape(a_hat)[0]-1), (0, np.shape(a_hat)[1]-1)], 'constant')

    if dim == 3:
        a_hat[-1, :,:]=a_hat[-1,:,:]/2
        a_hat[:,-1,  :]=a_hat[:,-1, :]/2
        a_hat[:, :, -1]=a_hat[:, :, -1]/2
        a_ext=np.pad(a_hat, [(0, np.shape(a_hat)[0]-1),(0, np.shape(a_hat)[1]-1),(0, np.shape(a_hat)[2]-1)], 'constant')

    return a_ext

def func2(x):
    return (x**2)+3

def tcheby_Basis(x,k):
    result=np.cos(k*np.arccos(x))
    return result


def tcheby_lagrange(x, i, N_n):

    x_i = cheb_extrema_grid1D(N_n)[i]

    T_k_xi=np.zeros(N_n)
    for k in range(N_n):
        T_k_xi[k]=tcheby_Basis(x_i, k)

    suma=np.zeros(1)
    for k in range(N_n):
        if k==0 or k== (N_n-1):
            suma = suma + (T_k_xi[k]*tcheby_Basis(x, k))/2
        else:
            suma = suma + T_k_xi[k]*tcheby_Basis(x, k)

    suma = 2*suma/(N_n-1)
    if i == 0 or i == N_n-1:
        suma = suma/2

    return suma


def DCT_normed(a):# this is exactly forward discrete transform u_hat(k)=2/(N+1) sum_i=0^N 1/c_i cos(kipi/N) u(i)
    a=dct(a, type = 1)/(np.size(a)-1)

    return a

def iDCT_normed(a):
    a=np.divide(a,2)

    return idct(a, type = 1)


def show_dirac_delta(N_n):
    for i in range(N_n):
        x =np.arange(-1,1.02,0.02)

        Lx=np.zeros(len(x))
        for j in range(len(x)):
            Lx[j]=tcheby_lagrange(x[j], i, N_n)

        plt.plot(x,Lx,label='Cheby_Lagrange {}'.format(i))
        plt.legend()
    plt.show()


def lagrange_weights(N_m):

    x = np.array(cheb_extrema_grid1D(N_m))
    field = np.arange(-1, 1.01, 0.01)
    integrale = np.zeros([N_m,N_m])

    for xi, yi in itertools.product(np.arange(0,N_m), np.arange(0,N_m)):
        y = np.zeros(np.size(x))
        y[xi] = 1
        lag_polx = lagrange(x, y)

        y2 = np.zeros(np.size(x))
        y2[yi] = 1
        lag_poly = lagrange(x, y2)

        discrete_x = lag_polx(field)
        discrete_y = lag_poly(field)

        discrete = np.outer(discrete_x, discrete_y)
        int = discrete.sum()/(np.size(discrete))
        #print(int)
        integrale[xi,yi]=int

    #print(integrale)



    return integrale


#########################################3
if __name__ == '__main__':
    N_n = 6
    u=np.array([0,0.125,0.06598301,0.125, 1.18401699,2])
    print('u')
    print(u)
    dCT=build_DCT_text_1D(N_n)
    print('dCT')
    print(dCT)
    print('Dctu')
    du=np.matmul(dCT, u)
    print(du)

  #  print('FFTPACK (Dctu)/N')
  #  print(dct(u, type = 1)/N_n)
  #  print('FFTPACK dct ortho')
  #  print(dct(u, type = 1,norm='ortho'))
  #  print(idct(dct(u, type = 1,norm='ortho'), type = 1, norm = 'ortho'))
    dCTt=np.transpose(dCT)
    print('DctT')
    print(dCTt)
    print('uDctT')
    print(np.matmul( u,dCTt))
    print('DcTu')
    print(np.matmul( dCTt,u))

    print('u')
    print(u)
    print('scipy(dct(u)')
    print(dctn_(u))
    print('scipy(dct_trans(u)')
    a=dctn_trans_(u)

    print(a)
    idCT = build_iDCT_text_1D(N_n)
    Ta=np.matmul(idCT, a)
    print(Ta)
    print(idctn_(a))
    print(np.matmul(idCT.transpose(), a))
    print('end')


    '''
    dct_orto=build_DCT_ortho_1D(N_n)
    print('dct ortho')
    print(dct_orto)
    print('dct ortho transposed')
    print(dct_orto.transpose())
    print('OThor u')
    print(np.matmul(dct_orto, u))



    idCT = build_iDCT_text_1D(N_n)
  #  print('idct')
  #  print(idCT)
    print('idctu')
    print(np.matmul(idCT, u))

    print('idctu_alg')
    print(idct(u/2,type = 1))

    print('DCTT algorithm my')
    u[0]=u[0]*2
    u[-1] = u[-1]*2

    newdu=np.matmul(dCT, u)
    FTPdctdu=idct(u/2, type = 1)

    newdu[0] = newdu[0]/2
    newdu[-1] = newdu[-1]/2
    FTPdctdu[0] = FTPdctdu[0]/2
    FTPdctdu[-1] = FTPdctdu[-1]/2
    print(newdu)
    print('DCTT alg fftpack')

    print(FTPdctdu)





    #I=np.matmul(dCT,dCTt)
   # print(I)
    '''
    '''
    sole=integrate_lagrange(N_n)
    breakpoint()
    x = np.array( cheb_extrema_grid1D(N_n))
    y = np.zeros(np.size(x))
    y[0]=1
    lag_polx= integrate_lagrange(x, y)

    y2 = np.zeros(np.size(x))
    y2[0] = 1
    lag_poly = integrate_lagrange(x, y2)

    field=np.arange(-1,1.01,0.01)
    discrete_x=lag_polx(field)
    discrete_y = lag_poly(field)

    discrete=np.outer(discrete_x,discrete_y)
    plt.plot(x, y,'o')
    plt.plot(field,discrete_x)
    plt.show()

    int=discrete.sum()/(np.size(discrete))
    print(int)
    '''

    '''
    d=3
    forma =d*[slice(None, None)]

    a=slice(-2, None, -2)

    for k in range(d):
        index=copy.deepcopy(forma)
        index[k] = a
        print(index)





    N_n = 9
    show_dirac_delta(N_n)
    x=cheb_extrema_grid1D(N_n)
    x2=cheb_extrema_grid1D(2*N_n-1)


    print(x)
    print(x2)
    

 #   dimension=2
  #  numNodes=4
  #  order=3
 #   N=np.array([5])
  #  poly1darraylist=np.empty(dimension, np)

    #vfun=np.vectorize(f_x)
   # a=np.random.rand(4)


   # u_n=idct_wiki([1,0,0,0,0])#np.arange(N_n)


   #  for k in range(N_n-1):
 #      print( tcheby_Basis(x_k, k))

  #  for k in range(N_n-1):
   #     print( sum(tcheby_Basis(x_k, k)))

    #

 #   vfun = np.vectorize(np.cos(5*x_k))

   # DCT = build_DCT_extrema1D_new(N_n)

 #   spectra=[1, 0, 1, 0,0]
 #   bb = iDCT_normed(spectra)
 #   dd = DCT_normed(bb)

 #   dd=dct(spectra, type = 1)
 #   aa = idct(spectra, type = 1)
 #   u_tilda = DCT.dot(spectra)

  #  aa = idct(spectra, type = 1)
  #  bb = iDCT_normed(spectra)

 #   cc=dct(bb, type = 1)

 #   x = cheb_extrema_grid1D(N_n+1)
 #   fx=f_x(x)


   # a=np.ones([4,4,3])
    #fa=f_x(a)
    AA=cheb_extrema_grid1D(7)
    at=nd_DCT(a)
    atx=nd_extend(at)
##    a_new=nd_iDCT(atx)
#    print (a)
#
#    print (at)
   # print(atx)

 #   print(a_new)
#
    '''
#########################
