"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Chem.Cheminfo    import DMCI
from admetnet.Chem.Atom        import DMAtom
from admetnet.Chem.Elements    import *
from admetnet.Utils.Status     import *
from admetnet.Utils.Exceptions import *

class DMMol(DMCI):
   """
   ADMETNet Molecule Object

   Attributes
   ----------
   
   name:          Name of the molecule
   formula:       Chemical formula
   in_train_set:  True if DMMol is in the Training set
   properties:    A list of DeepChem::Property
   atoms:         A list of DeepChem::Atom 
   """   
   def __init__(self, 
                name:str=None, 
                formula:str=None, 
                in_train_set:bool=True):

      DMCI.__init__(self)

      self._molname       = name
      self._formula       = formula
      self._in_train_set  = in_train_set
      self._properties    = list()
      self._atoms         = list()

      self._wfn = None
      self._qm_energy = None

   def __eq__(self, other):
      return (self.name == other.name and
              self.smiles == other.smiles)

   @property
   def wave_function(self):
      return self._wfn

   @wave_function.setter
   def wave_function(self, wfn):
      self._wfn = wfn

   @property
   def qm_energy(self):
      return self._qm_energy

   @qm_energy.setter
   def qm_energy(self, energy):
      self._qm_energy = energy

   @property
   def name(self)->str:
      return self._molname
   
   @name.setter
   def name(self, name:str):
      self._molname = name

   @property
   def formula(self)->str:
      return self._formula

   @formula.setter
   def formula(self, formula:str):
      self._formula = formula

   def is_intrain_set(self)->bool:
      return self._in_train_set

   def set_train_set(self, in_train_set:bool):
      self._in_train_set = in_train_set

   def add_atom(self, atom):
      """
      Add a new atom
      """
      try:
         if atom not in self._atoms:
            self._atoms.append(atom)
         else:
            raise AtomExists
      except AtomExists:
         Status.AtomExists(self._molname)

   def get_atoms(self):
      """
      Return a list of all atoms. 
      """
      return self._atoms 

   def get_atom(self, index):
      """
      Return atom by index
      """
      try:
         for atom in self._atoms:
            if atom.index() == index:
               return atom
         raise OutOfAtomIndex
      except OutOfAtomIndex:
         Status.OutOfAtomIndex(self._molname, index)

   def gen_atoms(self):
      self.set_atomtypes()
      for i in range(self._obMol.NumAtoms()):
         obAtom = self._obMol.GetAtom(i+1)
         dmAtom = DMAtom(obAtom.GetIndex(),
                         obAtom.GetAtomicNum(),
                         atomnum2element(obAtom.GetAtomicNum()),
                         obAtom.GetData("FFAtomType").GetValue(),
                         obAtom.GetX(),    
                         obAtom.GetY(),    
                         obAtom.GetZ())    
         self.add_atom(dmAtom)

   def set_atomprops(self):
      for atom in self._atoms:
         pass
         #print(atom.atom_type, atom.x, atom.y, atom.z)

   def get_props(self):
      """
      Return a list of all properties
      """
      return self._properties 

   def append_prop(self, prop):
      """
      Append a new property
      """
      try:
         if prop not in self._properties:
            self._properties.append(prop)
         else:
            raise PropExists
      except PropExists:
         Status.PropExists(self._molname, prop)

   def get_prop(self, prop_name):
      """
      Return one propety by name.
      """
      try:
         for prop in self._properties:
            if prop.get_name() == prop_name:
               return prop
         raise PropDoseNotExist
      except PropDoseNotExist:
         Status.PropDoseNotExist(self._molname, prop_name)

   def has_prop(self, prop_name):
      """
      Return True if a molecule has a property
      """
      for prop in self._properties:
         if prop.get_name() == prop_name:
            return True
      return False

   def clear_props(self):
      """
      Clear the list of molecular propeties
      """
      self._properties.clear()   

   def remove_prop(self, prop_name):
      """
      Remove a property from the list of properties 
      """
      try:
         prop = self.has_prop(prop_name)
         if prop:
            self._properties.remove(prop)
         else:
            raise PropDoseNotExist
      except PropDoseNotExist:
         Status.PropDoseNotExist(self._molname, prop_name)

   def graph_features(self,
                      features=None):
      """
      This function generates a graph for a sample
      molecule, e.g. benzene, and returns the number 
      of atom and bond features of the graph.
      """
      self.smiles = "c1ccccc1"
      self.rdMol_from_smiles()
      self.molgraph(features=features)
      nodes, edges = self.graph_node_edge_attr_matrix()
      _, num_atom_feature = nodes.shape
      _, num_bond_feature = edges.shape
      return num_atom_feature, num_bond_feature
