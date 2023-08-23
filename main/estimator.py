
from admetnet.ML.Networks       import *
from admetnet.Data.InputLoader  import *
from admetnet.Utils.Imports     import *

class BaseEstimator:
   """
   Class for BaseEstimator
   """
   def __init__(self, 
                network_type="GCN", 
                n_estimators=30):

      self._network_type = network_type
      self._n_estimators = n_estimators
      self._estimators   = []
      self._X            = []

   @property
   def network_type(self):
      return self._network_type

   @network_type.setter
   def network_type(self, network_type):
      self._network_type = network_type

   @property
   def n_estimators(self):
      return self._n_estimators

   @n_estimators.setter
   def n_estimators(self, n):
      self._n_estimators = n

   def load_estimators(self, path):
      for n in range(self._n_estimators):
         modelfile = path + "Model_{}.h5py".format(n)
         if os.path.exists(modelfile):
            estimator = Networks()
            estimator.set_modelfile(modelfile)
            estimator.network_type = self._network_type
            estimator.load_model()
            self._estimators.append(estimator)
         else:
            return False

      if self._estimators:
         return True
      else:
         return False

class Estimator(DMParser, BaseEstimator):
   def __init__(self, 
                inputfile=None,
                featurizer=None):

      BaseEstimator.__init__(self)
      DMParser.__init__(self)

      self._inputfile  = inputfile
      self._smiles     = None
      self._molecules  = []
      self._nmol       = 0

   @property
   def inputfile(self):
      return self._inputfile

   @inputfile.setter
   def inputfile(self, inputfile):
      self._inputfile = inputfile

   @property
   def smiles(self):
      return self._smiles

   @smiles.setter
   def smiles(self, smiles):
      self._smiles = smiles
      molecule = DMMol()
      molecule.name = "UNK"
      molecule.smiles = smiles
      self._molecules.append(molecule)
      self._nmol += 1

   def parse_file(self):
      csv = open(self._inputfile, "r").readlines()
      for i, row in enumerate(csv):
         columns = row.rstrip().split(",")
         ncols = len(columns)
         molecule = DMMol()
         if ncols == 1:
            molecule.name   = str(i)
            molecule.smiles = columns[0]
         elif ncols == 2:
            molecule.name   = columns[0]
            molecule.smiles = columns[1]
         else:
            sys.exit("Input .csv file must have one or two columns")

         self._molecules.append(molecule)
         self._nmol += 1

      if self._molecules:
         return True
      else:
         return False

   def X_generator(self)->bool:
      
      self.read_database_params()

      for molecule in self._molecules:
         if not molecule.rdMol_from_smiles(add_hydrogen=self._explicit_hydrogen):
            return False

      if self._featurizer == Featurizer.MolGraph:
         self.molgraph_featurizer()

      else:
         self.fingerprint_featurizer()

      return True

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

   def make_prediction(self):
      try:
         self._y = np.zeros((self._n_estimators, self._nmol))
         for i, estimator in enumerate(self._estimators):
            y = estimator.predict(self._X)
            for j in range(self._nmol):
               self._y[i][j] = y[j]
         return True
      except:
         return False

   def molecules(self):
      return self._molecules

   def stats(self):
      mean = np.mean(self._y.T, axis=1)
      std  = np.std(self._y.T, axis=1)
      return (mean, std)

def config_args(args):

   if args.input_file and args.input_smiles:
      sys.exit("You should upload either an input file or a SMILES string")

   if not args.input_file and not args.input_smiles:
      sys.exit("Please provide either an input file or a SMILES string")

   config = {}
   config["Network"]           = args.network_type
   config["N_Estimator"]       = args.n_estimator
   config["DATBASE_PARAMFILE"] = args.database_paramfile

   if args.input_file:
      config["InputFile"] = args.input_file
      config["SMILES"]    = None
   else:
      config["InputFile"] = None
      config["SMILES"]    = args.input_smiles

   return config 

def estimator(config):

   est = Estimator()

   est.network_type = config["Network"]
   est.n_estimators = config["N_Estimator"]
   est.set_database_paramfile(config["DATBASE_PARAMFILE"])

   if config["InputFile"]:
      est.inputfile = config["InputFile"]
      est.parse_file()
   else:
      est.smiles    = config["SMILES"]

   est.load_estimators(path=config["Estimators_PATH"])

   if est.X_generator():
      if est.make_prediction():
         molecules = est.molecules()
         mean, std = est.stats()
         return (molecules, mean, std)
      else:
         return None
   return None


