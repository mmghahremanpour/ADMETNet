"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Utils.Lambdas import *


class DMProp():
   """
   ADMETNet Property Object

   In ADMETNet atomic and molecular properties are 
   considered as tensors. For example, dipole is a 
   tensor of rank 1 whose norm yields the total dipole
   moment that is a tensor of rank 0.

   Attributes
   ----------

   name:       Property name
   source:     Source of the property value, e.g experiments
   value:      Property value as numpy.ndarray
   rank:       Rank of the property tensor
   trainable:  True if this property can be trained 
   """
   def __init__(self, 
                source:str=None, 
                name:str=None, 
                value:float=None,
                rank:int=0,
                trainable:bool=True):

      self._name:str       = name
      self._source:str     = source
      self._rank:int       = rank
      self._value          = nparf32(value)
      self._trainable:bool = trainable

   def __eq__(self, other):
      return (self._name   == other.get_name()   and
              self._source == other.get_source() and
              nparc(self._value, other.get_value()))

   def get_name(self):
      return self._name

   def set_name(self, name):
      self._name = name

   def get_source(self):
      return self._source

   def set_source(self, source):
      self._source = source

   def get_rank(self):
      return self._rank

   def set_rank(self, rank):
      self._rank = rank

   def get_value(self):
      return self._value

   def set_value(self, value):
      self._value = npar32(value)

   def istrainable(self):
      return self._trainable

   def set_trainable(self, trainable):
      self._trainable = trainable
