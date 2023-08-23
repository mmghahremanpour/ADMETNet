

from admetnet.Utils.Exceptions import *
from admetnet.Utils.Imports    import *
from admetnet.Utils.Status     import *
from admetnet.Chem.MolGraph    import *


class DMCI:
   """
   ADMETNet-Chemoinformatics Interface

   Attributes:
   ----------
   smiles:  SMILES Molecule Format
   smarts:  SMARTS Molecule Format
   rdMol:   RDKit Molecule 
   obMol:   OpenBabel Molecule
   dmGraph: ADMETNet Molecular Graph
   SDFfile: Spatial Data File (SDF) format
   acfp:    Atomi-Center Fingerprint
   apfp:    Atom-Pair Fingerprint
   mfp:     Morgan Fingerprint
   ttfp:    Topological-Torsion Fingerprint
   """
   def __init__(self):

      self._smiles:str = None
      self._smarts:str = None
      self._rdMol   = None
      self._obMol   = None
      self._DMGraph = None
      self._SDFfile = None

      self._acfp    = list()
      self._apfp    = None
      self._mfp     = None
      self._ttfp    = None

      self._atomic_descriptors = None
 
   def rdMol(self):
      return self._rdMol

   def obMol(self):
      return self._obMol

   def num_atoms(self):
      if self._rdMol:
         return self._rdMol.GetNumAtoms()
      elif self._obMol:
         return self._obMol.NumAtoms()
      else:
         return None

   def dmGraph(self):
      return self._DMGraph

   def atom_center_fingerprint(self):
      return self._acfp

   def atom_center_fingerprint_asArray(self):
      arr = np.zeros(1,)
      DataStructs.ConvertToNumpyArray(self._acfp, arr)
      return arr

   def atom_pair_fingerprint(self):
      return self._apfp

   def atom_pair_fingerprint_asArray(self):
      arr = np.zeros(1,)
      DataStructs.ConvertToNumpyArray(self._apfp, arr)
      return arr

   def morgan_fingerprint(self):
      return self._mfp

   def morgan_fingerprint_asArray(self):
      arr = np.zeros(1,)
      DataStructs.ConvertToNumpyArray(self._mfp, arr)
      return arr

   def topological_torsion_fingerperint(self):
      return self._ttfp

   def topological_torsion_fingerperint_asArray(self):
      arr = np.zeros(1,)
      DataStructs.ConvertToNumpyArray(self._ttfp, arr)
      return arr

   @property
   def SDFfile(self)->str:
      return self._SDFfile

   @SDFfile.setter
   def SDFfile(self, SDFfile:str):
      self._SDFfile = SDFfile

   @property
   def smiles(self)->str:
      return self._smiles

   @smiles.setter
   def smiles(self, smiles:str):
     if not isinstance(smiles, str):
       raise ValueError("SMILES must be string") 
     self._smiles = smiles 

   @property
   def smarts(self)->str:
      return self._samrts

   @smarts.setter
   def smarts(self, smarts:str):
      if not isinstance(smarts, str):
        raise ValueError("SMARTS must be string") 
      self._smarts = smarts

   def rdMol_from_smiles(self, add_hydrogen:bool=False)->bool:
      """
      Generates rdkit molecule object from SMILES string.
      """
      try:
         if self._smiles:
            self._rdMol = Chem.MolFromSmiles(self._smiles)

            if not self._rdMol:
               return False            

            if add_hydrogen:
               self._rdMol = Chem.AddHs(self._rdMol)

            self._rdMol = Chem.rdmolops.RenumberAtoms(self._rdMol, 
                          Chem.rdmolfiles.CanonicalRankAtoms(self._rdMol))
            return True
         else:
            raise MoleculeWithNoSMILE
      except MoleculeWithNoSMILE:
         Status.MoleculeWithNoSMILE()

   def rdMol_from_structure(self):
      if Path(self._SDFfile).suffix == ".sdf":
         self._rdMol = Chem.SDMolSupplier(self._SDFfile)
      else:
         Status.SDFfileNeeded()
   
   def rdMol_to_Psi4_XYZ(self)->str:
      AllChem.EmbedMolecule(self._rdMol, 
                            useExpTorsionAnglePrefs=True,
                            useBasicKnowledge=True)

      AllChem.UFFOptimizeMolecule(self._rdMol)
      xyz = "\n"
      for atom in self._rdMol.GetAtoms():
         pos = self._rdMol.GetConformer().GetAtomPosition(atom.GetIdx())
         xyz += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
      xyz += "units angstrom\n"
      return xyz

   def obMol_from_smiles(self, add_hydrogen:bool=False)->bool:
      """
      Generates OpenBabel obMol object from SMILES string.
      """
      try:
         if self._smiles:
            obConversion = openbabel.OBConversion()
            obConversion.SetInFormat("smi")
            self._obMol  = openbabel.OBMol()
   
            if not self._obMol:
               return False

            obConversion.ReadString(self._obMol, self._smiles)
            if add_hydrogen:
               self._obMol.AddHydrogens()

            builder = openbabel.OBBuilder()

            if not builder:
               return False

            builder.Build(self._obMol)

            return True

         else:
            raise MoleculeWithNoSMILE
      except MoleculeWithNoSMILE:
         Status.MoleculeWithNoSMILE()

   def _atom_pair_fingerprint(self,
                              nBits:int=512,
                              asBitVect:bool=True,
                              useChirality:bool=False, 
                              maxLength:int=4):
      """
      Returns atom-pair fingerprint for a molecule.
      """
      try:
         if self._rdMol:
            if asBitVect:
               self._apfp = AllChem.GetHashedAtomPairFingerprintAsBitVect(self._rdMol,
                                                                          nBits=nBits,
                                                                          maxLength=maxLength,
                                                                          includeChirality=useChirality)
            else:
               self._apfp = AllChem.GetHashedAtomPairFingerprint(self._rdMol, 
                                                                 maxLength=maxLength, 
                                                                 includeChirality=useChirality)
         else:
            raise RdKitMolNotGenerated
      except RdKitMolNotGenerated:
         Status.RdKitMolNotGenerated()

   def _atom_center_fingerprint(self, 
                                nBits:int=512, 
                                useChirality:bool=False,
                                maxLength:int=4):
      """
      Generates Atom-Pairs fingerprint for each 
      atom of a molecule. It returns an a list 
      np.array of fingerprints generated for 
      each atom.  
      """
      try:
         if self._rdMol:
            for atom in self._rdMol.GetAtoms():
               idx = atom.GetIdx()
               fp  = AllChem.GetHashedAtomPairFingerprintAsBitVect(self._rdMol, 
                                                                   nBits=nBits,
                                                                   maxLength=maxLength,
                                                                   includeChirality=useChirality,
                                                                   fromAtoms=[idx])
               arr = np.zeros(1,)
               DataStructs.ConvertToNumpyArray(fp, arr)
               self._acfp.append(arr)
         else:
            raise RdKitMolNotGenerated
      except RdKitMolNotGenerated:
         Status.RdKitMolNotGenerated()
   
   def _morgan_fingerprint(self, 
                          radius=2, 
                          nBits=512, 
                          useFeatures:bool=False, 
                          useChirality:bool=False, 
                          asBitVect:bool=True):
      """
      Returns Morgan fingerprint as BitVect.

      Features that can be used in the Morgan fingerprint:
      ---------------------------------------------------
         1) Hydrogen bond donor
         2) Hydrogen bond acceptor
         3) Aromatic
         4) Halogen
         5) Basic
         6) Acidic 
         
         These features will be considered if useFeatures is True.
      """
      try:
         if self._rdMol:
            if asBitVect:
               self._mfp = AllChem.GetMorganFingerprintAsBitVect(self._rdMol, 
                                                                 radius=radius, 
                                                                 nBits=nBits,
                                                                 useFeatures=useFeatures,
                                                                 useChirality=useChirality)
            else:
               self._mfp = AllChem.GetMorganFingerprint(self._rdMol,
                                                        radius=radius,
                                                        useFeatures=useFeatures,
                                                        useChirality=useChirality)
         else:
            raise RdKitMolNotGenerated
      except RdKitMolNotGenerated:
         Status.RdKitMolNotGenerated()

   def _toplogical_torsion_fingerprint(self, 
                                       nBits:int=2048, 
                                       asBitVect:bool=True, 
                                       useChirality:bool=False):
      """
      Returns Toplogical-Torsion fingerprint as BitVect. 
      """
      try:
         if self._rdMol:
            if asBitVect:
               self._ttfp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(self._rdMol, 
                                                                                    nBits=nBits,
                                                                                    includeChirality=useChirality)
            else:
               self._ttfp = AllChem.GetTopologicalTorsionFingerprint(self._rdMol,
                                                                     includeChirality=useChirality)
         else:
            raise RdKitMolNotGenerated
      except RdKitMolNotGenerated:
         Status.RdKitMolNotGenerated()

   def set_fingerprints(self, 
                        asBitVect:bool=True, 
                        nBits:int=2048, 
                        useChirality:bool=True, 
                        useFeatures:bool=True, 
                        maxLength:int=4,
                        radius:int=2):

      self._morgan_fingerprint(radius=radius,
                               nBits=nBits,
                               useFeatures=useFeatures,
                               useChirality=useChirality,
                               asBitVect=asBitVect)

      self._atom_center_fingerprint(nBits=nBits, 
                                    useChirality=useChirality,
                                    maxLength=maxLength)

      self._atom_pair_fingerprint(nBits=nBits,
                                  asBitVect=asBitVect,
                                  useChirality=useChirality, 
                                  maxLength=maxLength)

      self._toplogical_torsion_fingerprint(nBits=nBits, 
                                           useChirality=useChirality)

   def set_atomtypes(self, forcefield:str="gaff", rotor_search:bool=True):
      """
      This function add force field atom types to
      the Open Babel OBMol object. 
      """
      try:
         if self._obMol:
            ff = openbabel.OBForceField.FindForceField(forcefield)
            try:
               if ff:
                  ff.Setup(self._obMol)
                  ff.GetAtomTypes(self._obMol)
                  if rotor_search:
                     ff.SystematicRotorSearch(100)
                     ff.UpdateCoordinates(self._obMol) 
               else:
                  raise ForceFieldNotFound
            except ForceFieldNotFound:
               Status.ForceFieldNotFound(forcefield)
         else:
            raise OBMolNotGenerated
      except OBMolNotGenerated:
         Status.OBMolNotGenerated()

   def get_atomtype(self, idx:int=None)->str:
      atom  = self_obMol.GetAtom(idx)
      atype = atom.GetData("FFAtomType")     
      return atype.GetValue()

   def _get_atom_specific_descriptors(self):
      self._atomic_descriptors = [[] for i in self._rdMol.GetAtoms()]
      for i, descriptor in enumerate(Chem.rdMolDescriptors._CalcCrippenContribs(self._rdMol)):
         self._atomic_descriptors[i].append(descriptor[0]) # Contrbution to logP
         self._atomic_descriptors[i].append(descriptor[1]) # Contribution to NMR
      for i, descriptor in enumerate(Chem.rdMolDescriptors._CalcTPSAContribs(self._rdMol)):
         self._atomic_descriptors[i].append(descriptor)    # Contribution to Total Polar Surface Area
      for i, descriptor in enumerate(Chem.rdMolDescriptors._CalcLabuteASAContribs(self._rdMol)[0]):
         self._atomic_descriptors[i].append(descriptor)    # Contribution to Solvent Accessible Area
      #for i, descriptor in enumerate(EState.EStateIndices(self._rdMol)):
      #   print(descriptor)
      #   self._atomic_descriptors[i].append(descriptor)
   
   def molgraph(self, 
                features=None, 
                sort_nodes_by_degree:bool=True):
      """
      Generates an undirected attributed graph for a molecule.
      """ 
      if features is None:
         features = dict()
         features["atom_descriptors"] = False
         features["atomic_number"]    = True
         features["atomic_hybridization"] = True
         features["atom_degree"] = True
         features["implicit_valence"] = True
         features["number_of_hydrogens"] = True
         features["formal_charge"] = True
         features["ring_size"] = True
         features["aromaticity"] = True
         features["bond_order"] = True
         features["chirality"] = True
         features["bond_conjugation"] = True


      #Construct a MolGraph object 
      self._DMGraph = MolGraph()

      #Make the nodes of the MolGraph
      if features["atom_descriptors"]:
         self._get_atom_specific_descriptors()

      for i, rdAtom in enumerate(self._rdMol.GetAtoms()):
         node = Node()
         node.index = rdAtom.GetIdx()

         if features["atom_descriptors"]:
            node.atom_attributes(rdAtom=rdAtom, 
                                 features=features, 
                                 atom_descriptors=self._atomic_descriptors[i])
         else:
            node.atom_attributes(rdAtom=rdAtom, 
                                 features=features)

         for neighbor in rdAtom.GetNeighbors():
            rdBond = self._rdMol.GetBondBetweenAtoms(rdAtom.GetIdx(), neighbor.GetIdx())
            node.append_neighbors(neighbor.GetIdx(), rdBond.GetIdx()) 

         self._DMGraph.append_node(node)

      #Make the edges of the MolGraph
      for rdBond in self._rdMol.GetBonds():
         edge = Edge()
         edge.index = rdBond.GetIdx()

         edge.bond_attributes(rdBond=rdBond, 
                              features=features)

         edge.append_connects(rdBond.GetBeginAtomIdx(), 
                              rdBond.GetEndAtomIdx())

         self._DMGraph.append_edge(edge)

      if sort_nodes_by_degree:
         self._DMGraph.sort_nodes_by_degree()
      
      return self._DMGraph


   def graph_attr_tensor(self):
      """
      Return the attribute matrix of a Moelcular Graph

      attribute_matrix: shape(N, Fn+Fe)
            N: Number of nodes (atoms)
            Fn: Number of features of each node
            Fe: Number of features of each edge
      """
      if self._DMGraph:
         self._DMGraph.make_graph_attr_tensor()
         return self._DMGraph.graph_attr_tensor()
      else:
         return None

   def graph_adj_matrix(self, max_num_node:int=None, add_edge_attr=False):
      """
      Returns the adjacency matrix of a Moelcular Graph

      The shape of adjacency_matrix is (N, N), where N is
      the number of nodes (atoms)
      """

      if self._DMGraph:
         self._DMGraph.make_adjacency_matrix(max_num_node=max_num_node, add_edge_attr=add_edge_attr)
         return self._DMGraph.adj_matrix()
      else:
         return None

   def graph_laplacian_matrix(self, normalized=True):
      """
      Returns the laplacian matrix of a Molecular Graph. 
      """
      if self._DMGraph:
         if normalized:
            self._DMGraph.make_normalized_laplacian_matrix()
            return self._DMGraph.normalized_laplacian_matrix()
         else:
            self._DMGraph.make_laplacian_matrix()
            return self._DMGraph.laplacian_matrix()
      else:
         return None

   def graph_node_attr_adj_matrix(self, max_num_node:int=None, add_edge_attr=False):
      """
      Returns Node attribute of and adjacency matrix of a Molecular Graph

      returns (node_attribute_matrix, adjacency_matrix)

      """

      if self._DMGraph:
         self._DMGraph.make_node_edge_attr_matrix(max_num_node=max_num_node)
         self._DMGraph.make_adjacency_matrix(max_num_node=max_num_node, add_edge_attr=add_edge_attr)
         return self._DMGraph.node_attr_adj_matrix()
      else:
         return (None, None)

   def graph_node_edge_attr_matrix(self, max_num_node=None)->Tuple:
      """
      Returns Node and Edge attributes of a Molecular Graph

      returns (node_attribute_matrix, edge_attribute_matrix)

      node_attribute_matrix: shape(N, Fn)
            N: Number of nodes (atoms)
            Fn: Number of features of each node

      edge_attribute_matrix: shape(N, Fe)
            N: Number of nodes (atoms)
            Fe: Number of features of each edge
      """
      if self._DMGraph:
         self._DMGraph.make_node_edge_attr_matrix(max_num_node=max_num_node)
         return self._DMGraph.node_edge_attr_matrix()
      else:
         return (None, None)
