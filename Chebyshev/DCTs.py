from scipy.fftpack import dctn, idctn
import numpy as np
import copy


def dctn_(nod_values):
    factor = 1
    for d in range(len(nod_values.shape)):
        factor = factor*(nod_values.shape[d] - 1)

    spectrum = dctn(nod_values, type = 1)/(factor)

    return spectrum

def idctn_(spectrum):

    #spectrum = spectrum/(2**len(spectrum.shape))
    nod_values = idctn(spectrum, type = 1)
    nod_values=nod_values/(2**len(spectrum.shape))
    return nod_values

def dctn_trans_(nod_values):
    factor = 1
    for d in range(len(nod_values.shape)):
        factor = factor*(nod_values.shape[d] - 1)

   # nod_values = multiply_both_dir(nod_values,2,0)
    nod_values = multiply_both_sym(nod_values, 2)
    spectrum = dctn(nod_values, type = 1)
    spectrum =spectrum/(factor)
   # spectrum = multiply_both_dir(spectrum, 1/2,0)
    spectrum = multiply_both_sym(spectrum, 1/2)
   # print(spectrum)

    return spectrum

def idctn_trans_(spectrum):

   # spectrum = spectrum/(2**len(spectrum.shape))
    #spectrum   = multiply_both_dir(spectrum, 2,0)
    spectrum = multiply_both_sym(spectrum, 2)
    nod_values = dctn(spectrum, type = 1)
    nod_values=nod_values/(2**len(spectrum.shape))
   # nod_values = multiply_both_dir(nod_values, 1/2,0)
    nod_values = multiply_both_sym(nod_values, 1/2)
    return nod_values


def dctn_ortho_(nod_values):  ## TODO: not verified/ finnished
    spectrum = dctn(nod_values, type = 1, norm='ortho')
    return spectrum

def idctn_ortho_(spectrum):## TODO: not verified/ finnished
    nod_values = idctn(spectrum, type = 1,norm='ortho')
    return nod_values


def grad_(spectrum, dir):
    dim = np.size(np.shape(spectrum))
    forma = dim*[slice(None, None)]

    indexc = copy.deepcopy(forma)
    indexc[dir] = -1
    spectrum[tuple(indexc)] = spectrum[tuple(indexc)]/2

    indexd = dim*[np.newaxis]
    indexd[dir] = slice(None, None)
    spectrum[tuple(forma)] = spectrum[tuple(forma)]*np.arange(0, np.shape(spectrum)[dir])[tuple(indexd)]

    grad_spectrum = np.zeros(np.shape(spectrum))

    indexa = copy.deepcopy(forma)
    indexa[dir] = slice(-2, None, -2)
    indexb = copy.deepcopy(forma)
    indexb[dir] = slice(-3, None, -2)

    indexe = copy.deepcopy(forma)
    indexe[dir] = slice(-1, 0, -2)
    indexf = copy.deepcopy(forma)
    indexf[dir] = slice(-2, 0, -2)

    grad_spectrum[tuple(indexa)] = np.cumsum(spectrum[tuple(indexe)], axis = dir)
    grad_spectrum[tuple(indexb)] = np.cumsum(spectrum[tuple(indexf)], axis = dir)

    grad_spectrum = 2*grad_spectrum
   # grad_spectrum=divide_last(grad_spectrum)

    return grad_spectrum

def grad_ortho_(spectrum, dir):
    dim = np.size(np.shape(spectrum))
    forma = dim*[slice(None, None)]

    indexc = copy.deepcopy(forma)
    indexc[dir] = -1
    spectrum[tuple(indexc)] = spectrum[tuple(indexc)]/2

    indexd = dim*[np.newaxis]
    indexd[dir] = slice(None, None)
    spectrum[tuple(forma)] = spectrum[tuple(forma)]*np.arange(0, np.shape(spectrum)[dir])[tuple(indexd)]

    grad_spectrum = np.zeros(np.shape(spectrum))

    indexa = copy.deepcopy(forma)
    indexa[dir] = slice(-2, None, -2)
    indexb = copy.deepcopy(forma)
    indexb[dir] = slice(-3, None, -2)

    indexe = copy.deepcopy(forma)
    indexe[dir] = slice(-1, 0, -2)
    indexf = copy.deepcopy(forma)
    indexf[dir] = slice(-2, 0, -2)

    grad_spectrum[tuple(indexa)] = np.cumsum(spectrum[tuple(indexe)], axis = dir)
    grad_spectrum[tuple(indexb)] = np.cumsum(spectrum[tuple(indexf)], axis = dir)

    grad_spectrum = 2*grad_spectrum

#    grad_spectrum=divide_first(grad_spectrum)

    return grad_spectrum






def grad_adjoin_(spectrum, dir):
    dim = np.size(np.shape(spectrum))
    forma = dim*[slice(None, None)]

    grad_spectrum = np.zeros(np.shape(spectrum))

    # even indexes for output
    indexa = copy.deepcopy(forma)
    indexa[dir] = slice(2, None, 2)

    # odd indexes for input
    indexe = copy.deepcopy(forma)
    indexe[dir] = slice(1, -1, 2)

    # odd indexes for output
    indexb = copy.deepcopy(forma)
    indexb[dir] = slice(1, None, 2)

    # even indexes for input
    indexf = copy.deepcopy(forma)
    indexf[dir] = slice(0, -1, 2)


    grad_spectrum[tuple(indexa)] = np.cumsum(spectrum[tuple(indexe)], axis = dir)
    grad_spectrum[tuple(indexb)] = np.cumsum(spectrum[tuple(indexf)], axis = dir)
    # multiply with "frequency" coeficient k
    indexd = dim*[np.newaxis]
    indexd[dir] = slice(None, None)
    grad_spectrum[tuple(forma)] = grad_spectrum[tuple(forma)]*np.arange(0, np.shape(spectrum)[dir])[tuple(indexd)]

    # multiply coeficients by 2
    grad_spectrum = 2*grad_spectrum

    # divide last coeficients by 2
    indexc = copy.deepcopy(forma)
    indexc[dir] = -1
    grad_spectrum[tuple(indexc)] = grad_spectrum[tuple(indexc)]/2

    return grad_spectrum


def decrease_spectrum(spectrum, M):
    if np.less_equal(np.shape(spectrum), M).any():
        raise ValueError('M is higher than size of spectrum')

    slc = [slice(0, M[i], 1) for i in range(np.size(np.shape(spectrum)))]

    return multiply_last(spectrum[tuple(slc)])


def enlarge_spectrum(spectrum):
    dim = np.size(np.shape(spectrum))#spectrum.shape.__len__()
    divide_last(spectrum)
    if dim == 1:
        spectrum = np.pad(spectrum, [(0, np.shape(spectrum)[0] - 1)],
                          'constant')
    if dim == 2:
        spectrum = np.pad(spectrum, [(0, np.shape(spectrum)[0] - 1), (0, np.shape(spectrum)[1] - 1)],
                          'constant')
    if dim == 3:
        spectrum = np.pad(spectrum,
                          [(0, np.shape(spectrum)[0] - 1), (0, np.shape(spectrum)[1] - 1),
                           (0, np.shape(spectrum)[2] - 1)],
                          'constant')

    return spectrum


def divide_both(spectrum):
    divide_first(spectrum)
    divide_last(spectrum)

    return spectrum

def divide_both_sym(spectrum):
    divide_first_sym(spectrum)
    divide_last_sym(spectrum)

    return spectrum

def multiply_both_sym(spectrum,factor):
    multiply_first_sym(spectrum,factor)
    multiply_last_sym(spectrum,factor)

    return spectrum

def multiply_both_dir(spectrum,factor,dir):
    multiply_first_dir(spectrum,factor,dir)
    multiply_last_dir(spectrum,factor,dir)

    return spectrum

def divide_first(spectrum):
    dim = np.size(np.shape(spectrum))

    for d in range(dim):
        index = dim*[slice(None, None)]
        index[d] = 0
        spectrum[tuple(index)] = spectrum[tuple(index)]/2

    return spectrum

def divide_last(spectrum):
    dim = np.size(np.shape(spectrum))

    for d in range(dim):
        index = dim*[slice(None, None)]
        index[d] = -1
        spectrum[tuple(index)] = spectrum[tuple(index)]/2

    return spectrum

def divide_first_sym(spectrum):
    dim = np.size(np.shape(spectrum))

    for d in range(dim):
        index = dim*[slice(None, None)]
        index[d] = 0
        spectrum[tuple(index)] = spectrum[tuple(index)]/np.sqrt(2)

    return spectrum

def divide_last_sym(spectrum):
    dim = np.size(np.shape(spectrum))

    for d in range(dim):
        index = dim*[slice(None, None)]
        index[d] = -1
        spectrum[tuple(index)] = spectrum[tuple(index)]/np.sqrt(2)

    return spectrum

def multiply_last(spectrum):
    dim = np.size(np.shape(spectrum))

    for d in range(dim):
        index = dim*[slice(None, None)]
        index[d] = -1
        spectrum[tuple(index)] = spectrum[tuple(index)]*2

    return spectrum

def multiply_last_sym(spectrum,factor):
    dim = np.size(np.shape(spectrum))

    for d in range(dim):
        index = dim*[slice(None, None)]
        index[d] = -1
        spectrum[tuple(index)] = spectrum[tuple(index)]*factor

    return spectrum

def multiply_first_sym(spectrum,factor):
    dim = np.size(np.shape(spectrum))

    for d in range(dim):
        index = dim*[slice(None, None)]
        index[d] = 0
        spectrum[tuple(index)] = spectrum[tuple(index)]*factor

    return spectrum


def multiply_first_dir(spectrum,factor,dir):
    dim = np.size(np.shape(spectrum))

    #for d in range(dim):
    index = dim*[slice(None, None)]
    index[dir] = 0
    spectrum[tuple(index)] = spectrum[tuple(index)]*factor

    return spectrum


def multiply_last_dir(spectrum,factor,dir):
    dim = np.size(np.shape(spectrum))

    index = dim*[slice(None, None)]
    index[dir] = -1
    spectrum[tuple(index)] = spectrum[tuple(index)]*factor

    return spectrum


