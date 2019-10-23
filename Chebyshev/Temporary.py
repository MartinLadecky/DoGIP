
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



