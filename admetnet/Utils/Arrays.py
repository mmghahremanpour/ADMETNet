
from admetnet.Utils.Imports import *


def sample_size(a):

   message = "Expected Python list or numpy ndarray, got %s" % type(a)

   if isinstance(a, np.ndarray):
      if a.shape is not None:
         if len(a.shape) == 0:
            raise TypeError ("%r cannot be considered a valid collection." % a)
         else:
            return a.shape[0]

   elif isinstance(a, list):
      a = np.asarray(a)
      return a.shape[0]

   else:
      raise TypeError(message)

def check_array_consistency(*arrays):
   
   lengths = [sample_size(arr) for arr in arrays if arr is not None]
   uniques = np.unique(lengths)

   if len(uniques) > 1:
      raise LenghtError("Found two arrays with different lenght")
   else:
      return True

def arraylike(a):
   return isinstance(a, np.ndarray)

def make_arraylike(a):
   return np.asarray(a, dtype=np.ndarray)

