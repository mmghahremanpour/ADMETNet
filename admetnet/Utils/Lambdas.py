"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from .Imports import *


npari32   = lambda a: np.array(a, dtype=np.int32)            # numpy.ndarray of type Integer 
npari64   = lambda a: np.array(a, dtype=np.int64)
nparf32   = lambda a: np.array(a, dtype=np.float32)          # numpy.ndarray of type Float
nparf64   = lambda a: np.array(a, dtype=np.float64)

nparc     = lambda a,b: np.array_equal(a,b)  # numpy array comparison
ai        = lambda a,i: a[i]                 # Return element i of a 1D array
aij       = lambda a,i,j : a[i][j]           # Return element ij of a 2D array
aijk      = lambda a,i,j,k : a[i][j][k]      # Return element ijk of a 3D array

bound      = lambda value, array: array[-1] if value not in array else value
hotvec_unk = lambda value, array: map(lambda x: x==bound(value,array), array)
hotvec     = lambda value, array: map(lambda x: x==value, array)

cumsum     = lambda a, offset=0: np.insert(np.cumsum(a), 0, 0) + offset
cumsum_unk = lambda a, offset=0: np.delete(np.insert(np.cumsum(a), 0, 0), -1) + offset


