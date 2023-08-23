
from admetnet.Utils.Imports   import *
from admetnet.Utils.Enums     import *
from admetnet.Chem.Molecule   import *
from admetnet.Data.BaseParser import *


def path_to_string(path):
  if isinstance(path, os.PathLike):
    return os.fspath(path)
  return path

def ask_user_to_overwrite(filepath):
   overwrite = input("%s already exists! Do you want to overwrite? [y/n]"
                     % filepath).strip().lower()
   while overwrite not in ('y', 'n'):
      overwrite = input('Enter "y" (overwrite) or "n" ''(cancel).').strip().lower()
   if overwrite == 'n':
      return False
   return True

class InputParser(DMParser):

   def __init__(self):

      DMParser.__init__(self)

      self._properties = list()
      self._molecules  = list()
      self._X          = list()

      self._learning_method:str = None
      
   def gen_mols(self, add_hydrogen:bool=True)->bool:

      self.read_database_params()
      self.read_database()

      for key, value in self._db.items():
         if key == JsonKeys.Learning_Method.name:
            self.learning_method = value
         elif key == JsonKeys.Properties.name:
            for propname in value:
               self._properties.append(propname)
         elif key == JsonKeys.Molecules.name:
            for v in value:
               if v[JsonKeys.SMILES.name] is not None:
                  molecule = DMMol()
                  molecule.name   = v[JsonKeys.Molname.name]
                  molecule.smiles = v[JsonKeys.SMILES.name]
                  if (molecule.rdMol_from_smiles(add_hydrogen=self._explicit_hydrogen) and 
                      molecule.obMol_from_smiles(add_hydrogen=self._explicit_hydrogen)):

                     self._max_num_atom = max(self._max_num_atom, molecule.num_atoms())
                     self._molecules.append(molecule)

      if self._molecules:
         return True
      else:
         return False

   def X_generator(self)->NoReturn:

      if self._featurizer == Featurizer.MolGraph:
         self.molgraph_featurizer()

      else:
         self.fingerprint_featurizer()

   def molgraph_featurizer(self)->NoReturn:
      """
      Output is a tuple (node_attr, adj_matrix)
      """

      graph_features   = self.get_graph_features()
      num_node_attr, _ = DMMol().graph_features(features=graph_features)

      self._max_num_atom=None

      def generator():
         for molecule in self._molecules:
            molecule.molgraph(features=graph_features)
            X = molecule.graph_node_attr_adj_matrix(max_num_node=self._max_num_atom, 
                                                    add_edge_attr=self._edge_features)
            yield X, 0.0

      self._X = tf.data.Dataset.from_generator(generator, 
                                               output_types=((tf.float32, tf.float32), (tf.int32)), 
                                               output_shapes=((tf.TensorShape([None, num_node_attr]), 
                                               tf.TensorShape([None, None])), tf.TensorShape([])))

   def fingerprint_featurizer(self)->NoReturn:

      for molecule in self._molecules:

         molecule.set_fingerprints(asBitVect=self._fingerprint_asBitVec,
                                   nBits=self._fingerprint_nBits,
                                   useChirality=self._chirality,
                                   useFeatures=self._atom_descriptors,
                                   maxLength=self._atompair_fp_length,
                                   radius=self._morgan_fp_radius)

         feature = None
         if self._featurizer == Featurizer.Morgan_Fingerprint:
            feature = molecule.morgan_fingerprint_asArray()

         elif self._featurizer == Featurizer.Atom_Pair_Fingerprint:
            feature = molecule.atom_pair_fingerprint_asArray()

         elif self._featurizer == Featurizer.Atom_Center_Fingerprint:
            featuer = molecule.atom_center_fingerprint_asArray()

         elif self._featurizer == Featurizer.Topological_Torsion_Fingerprint:
            feature = molecule.topological_torsion_fingerperint_asArray()

         else:
            Status.NotSupportedFeaturizer(self._featurizer.name)
         self._X.append(feature)

      self._X = np.asarray(self._X)

   def parse_input(self)->bool:

      if self.gen_mols():
         self.X_generator()
         return True
      else:
         return False

   def get_X(self):
      return self._X

   def get_molecules(self):
      return self._molecules

   def get_properties(self):
      return self._properties

   def get_learning_method(self):
      return self._learning_method

