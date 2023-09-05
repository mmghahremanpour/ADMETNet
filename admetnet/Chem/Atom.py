"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Chem.Elements import *
from admetnet.Chem.Property import *

class DMAtom:
   """
   ADMETNet Atom Object
   """
   def __init__(self,
                index:int=None,
                atomnum:int=None,
                natomname:str=None,
                mass:float=None,
                atype:str=None,
                x:float=0.0, 
                y:float=0.0, 
                z:float=0.0):

      self._symbol  = None
      self._index   = index
      self._atomnum = atomnum
      self._atomname= atomname
      self._mass    = mass
      self._atype   = atype
      self._props   = list()

      self._x = x
      self._y = y
      self._z = z

   def __eq__(self, other):
      return (self._index == other.index)

   def get_index(self)->int:
      return self._index

   def set_index(self, idx:int=None):
      self._index = idx

   def get_atomnum(self)->int:
      return self._atomnum

   def set_atomnum(self, atomnum:int):
      self._atomnum = atomnum

   def set_atomname(self, atomname:str):
      self._atomname = atomname

   def get_atomname(self)->str:
      return self._atomname

   def get_mass(self)->float:
      return self._mass

   def set_mass(self, mass:float=None):
      self._mass = mass

   def get_atype(self)->str:
      return self._atype

   def set_atype(self, atype:str):
      self._atype = atype

   def get_x(self)->float:
      return self._x

   def set_x(self, x:float):
      self._x = float(x)

   def get_y(self)->float:
      return self._y

   def set_y(self, y:float):
      self._y = float(y)

   def get_z(self)->float:
      return self._z

   def set_z(self, z:float):
      self._z = float(z)

   def get_props(self):
      return self._props

   def set_props(self, prop):
      if prop not in self._props:
         self._props.append(prop) 

   def prop(self, prop_name):
      for prop in props:
         if prop.name() == prop_name:
            return prop

   def coordinate(self):
      return npar([self._x, self._y, self._z])

def rdAtom_to_dmAtom(rdAtom=None):
   if rdAtom:
      dmAtom = DMAtom()

      dmAtom.set_atomnum(rdAtom.getAtomicNum())
      dmAtom.set_symbol(rdAtom.getSymbol())
      dmAtom.set_index(rdAtom.getIdx())
      dmAtom.set_mass(atomnum2mass(rdAtom.getAtomicNum()))

