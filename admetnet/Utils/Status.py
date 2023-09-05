"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Utils.Imports import *

@unique
class Status(Enum):
   normal   = 1
   abnormal = 2

   PropValue_IsNone = 3
   MolProp_NotFound = 4
   DataSource_NotFound = 5
   SMILES_IsAbnormal = 6
  
   @classmethod 
   def thank_you_note(cls):
      print("\n%68s" % "Thank you for using ADMETNet")
      print("%65s"   % "Yale University (2021-)\n")
      print("%90s"   % "ADMETNet is free software under the Gnu Public License v 2.");
      print("%80s"   % "Read more at http://www.gnu.org/licenses/gpl-2.0.html");

   def print_line():
      print("\t\t==========================================\n")


   @classmethod
   def print_status(cls, status):
      print("\n       Status: %s.\n" % status.name) 
      cls.thank_you_note()
      sys.exit()
      

   @classmethod
   def MoleculeWithNoSMILE(cls):
      print("\n       ERROR!    ADMETNet needs a SMILE string to generate Rdkit MOL object.\n") 
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def RdKitMolNotGenerated(cls):
      print("\n       ERROR!    Rdkit MOL object is not generated!\n") 
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def OBMolNotGenerated(cls):  
      print("\n       ERROR!    OpenBabel OBMOL object is not generated!\n") 
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def PropExists(cls, molname, prop):
      print("\n       ERROR!    Trying to add %s to %s twise!\n" %(prop.name, molname))
      cls.thank_you_note()
      sys.exit()
       
   @classmethod
   def PropDoseNotExist(cls, molname, prop_name):
      print("\n       ERROR!    Trying to remove %s from %s that dose not exist!\n" %(prop_name, molname))
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def AtomExists(cls, molname):
      print("\n       ERROR!    Trying to add an atom with the same coordinates to %s twise!\n" %(molname))
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def OutOfAtomIndex(cls, molname, idx):
      print("\n       ERROR!    Atom index %s does not exist in %s!\n" %(idx, molname))
      cls.thank_you_note()
      sys.exit()
   
   @classmethod
   def ForceFieldNotFound(cls, forcefield):
      print("\n       ERROR!    %s force field not found in Open Babel!\n" % forcefield)
      cls.thank_you_note()
      sys.exit()
      
   @classmethod
   def NoMolculeInData(cls, datafile):
      print("\n       ERROR!    No molecule exists in %s!\n" % datafile)
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def NoSMILESInData(cls, datafile):
      print("\n       ERROR!    One or more molecule(s) do not have SMILES in %s!\n" % datafile)
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def NoIUPACInData(cls, datafile):
      print("\n       ERROR!    One or more molecule(s) do not have name in %s!\n" % datafile)
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def NoRefPropertyInData(cls, datafile):
      print("\n       ERROR!    One or more molecule(s) do not have reference property in %s!\n" % datafile)
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def NoTrainSet(cls):
      print("\n       ERROR!    Training probability is zero, so no molecule will be added to the training set\n")
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def NotSupportedFeaturizer(cls, featurizer):
      print("\n       ERROR!    ADMETNet dose not support %s as molecular featurizer!\n" % featurizer)
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def UnsupportedHyperParameter(cls, hp):
      print("\n       ERROR!    ADMETNet dose not support hyperparameter %s!\n" % hp)
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def UnsupportedOptimizer(cls, optimizer):
      print("\n       ERROR!    ADMETNet dose not support %s optimizer for training!\n" % optimizer)
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def CorruptMolGraph(cls):
      print("\n       ERROR!    A None MolGraph object is passed to ConvMolGraph class\n")
      cls.thank_you_note()
      sys.exit()
   
   @classmethod
   def ZeroConvLayer(cls):
      print("\n       ERROR!    You need to have at least 1 conv layer for GCNN network type\n")
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def WrongSizeForAttrMatrix(cls):
      print("\n       ERROR!    Shape of the attribute matrix does not match the number of nodes\n")
      cls.thank_you_note()
      sys.exit()
   
   @classmethod
   def NoEdgeforAdjMatrix(cls):
      print("\n       ERROR!    Cannot make adjacenecy matrix for a graph without any edge\n")
      cls.thank_you_note()
      sys.exit()

   @classmethod
   def SamllTrainSet(cls):
      print("\n       WARNING!    Training probability is lower than 0.2, so the size of the tarin set will be very small\n")
   
   @classmethod
   def SmallTestSet(cls):
      print("\n       WARNING!    Training probability is higher than 0.8, so the size of the test set will be very small\n")

      
