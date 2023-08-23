
from admetnet.Utils.Imports import *
from admetnet.Utils.Lambdas import *
from admetnet.Utils.Enums   import *
from admetnet.Utils.PyPlot  import *
from admetnet.Utils.Status  import *
from admetnet.Chem.Molecule import *
from admetnet.Chem.Property import *
from admetnet.Data.BaseParser import *
from admetnet.Data.Generator import *
from admetnet.Data.Molselect import *

class DataLoader(DMParser):
   """
   This calss load database
   """
   def __init__(self):

      DMParser.__init__(self) 

      self._molecules    = list()
      self._training_set = list()
      self._test_set     = list()


      self._X_train:List = list()
      self._y_train:List = list()
      self._X_test:List = list()
      self._y_test:List = list()

      self._X_All:List = list()
      self._y_All:List = list()

      self._dataset = None
      self._test_data = None
      self._val_data = None
      self._train_data = None 
         
   
   def gen_mols(self)->bool:

      self.read_database_params()
      self.read_database()

      if self.check_database():
         for key, value in self._db.items():
            if key == JsonKeys.Molecules.name:
               for v in value:
                  status = Status.normal
                  if v[JsonKeys.SMILES.name] is not None:
                     molecule = DMMol()
                     molecule.name = v[JsonKeys.Molname.name]
                     molecule.smiles = v[JsonKeys.SMILES.name]
                     if (molecule.rdMol_from_smiles(add_hydrogen=self._explicit_hydrogen) and
                         molecule.obMol_from_smiles(add_hydrogen=self._explicit_hydrogen)):

                        self._max_num_atom = max(self._max_num_atom, molecule.num_atoms())

                        if self._dbsource.name in v[JsonKeys.Properties.name].keys():
                           prop = v[JsonKeys.Properties.name][self._dbsource.name]
                           if self._propname in prop.keys(): 
                              prop_value = prop[self._propname]
                              if prop_value is not None:
                                 prop = DMProp(self._dbsource, self._propname, prop_value)
                                 molecule.append_prop(prop)
                              else:
                                 status = Status.PropValue_IsNone
                           else:
                              status = Status.MolProp_NotFound
                        else:
                           status = Status.DataSource_NotFound
                     else:
                        status = Status.SMILES_IsAbnormal

                     if status == Status.normal:
                        self._molecules.append(molecule)

                     else:
                        Status.print_status(status)
         return True
      else:
         return False

   def train_test_split(self)->NoReturn:
      """
      This function splits the database into 
      train and test sets. 
      """ 

      if self._molecules:
         for molecule in self._molecules:
            if np.random.uniform(0, 1) <= self._test_to_train_ratio:
               molecule.set_train_set(True)
               self._training_set.append(molecule)
            else:
               molecule.set_train_set(False) 
               self._test_set.append(molecule)
      else:
        Status.NoMolculeInData(self._db_file) 

   def X_generator(self)->NoReturn:
      """
      This function featurizes molecules
      """
      if self._featurizer == Featurizer.MolGraph:
         self.molgraph_featurizer()

      else:
         self.fingerprint_featurizer()


   def molgraph_featurizer(self)->NoReturn:
      """
      Output is a tuple ((node_attr, adj_matrix), y)
      """

      graph_features     = self.get_graph_features()
      n_node_attr, _     = DMMol().graph_features(features=graph_features)
      self._max_num_atom = None

      generator = Generator()
      generator.set_features(graph_features)
      generator.set_maxatom(self._max_num_atom)
      generator.set_edgeattr(self._edge_features)
      generator.set_n_node_attr(n_node_attr)
      generator.set_propname(self._propname)
      generator.set_labels(self._class_labels)

      molselect = Molselect()
      molselect.set_molset(self._molecules)
      molselect.set_shuffle(self._shuffle)
      molselect.set_bootstrap(self._bootstrap)
      molselect.set_ratio(self._test_to_train_ratio)

      self._training_set, self._test_set = molselect.make_selection()

      if self._class_labels:
         self._train_data = generator.classification(molset=self._training_set)
         self._test_data  = generator.classification(molset=self._test_set)
         self._val_data   = generator.classification(molset=self._test_set)
         self._dataset    = generator.classification(molset=self._molecules)
      else:
         self._dataset    = generator.regression(molset=self._molecules)
         """
         self._train_data = generator.regression(molset=self._training_set)
         self._train_data = self._train_data.take(sum(1 for _ in self._train_data))
         self._val_data   = generator.regression(molset=self._test_set)
         self._val_data   = self._val_data.take(sum(1 for _ in self._val_data))
         self._dataset    = self._val_data
         #self._train_data = self._val_data
         #self._test_data  = generator.regression(molset=self._test_set)
         self._test_data  = self._val_data
         """
         
         full_ds_size = sum(1 for _ in self._dataset) 
         train_ds_size = int(self._test_to_train_ratio * full_ds_size)
         valid_ds_size = int((1-self._test_to_train_ratio) * full_ds_size)

         self._train_data = self._dataset.take(train_ds_size)
         self._test_data = self._dataset.skip(valid_ds_size).take(valid_ds_size)
         self._val_data = self._dataset.skip(valid_ds_size).take(valid_ds_size)
         

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

         self._X_All.append(feature)

         if molecule.is_intrain_set():
            self._X_train.append(feature)
         else:
            self._X_test.append(feature)

      self._X_All = np.asarray(self._X_All)
      self._X_train = np.asarray(self._X_train)
      self._X_test = np.asarray(self._X_test)

   def y_generator(self, categorical:bool=False):
      if categorical:
         self.y_categorical_dataset()
      else:
         self.y_dataset()

   def y_dataset(self):
      for molecule in self._molecules:
         y = molecule.get_prop(self._propname) 
         y = float(y.get_value())
         
         self._y_All.append(y)

         if molecule.is_intrain_set():
            self._y_train.append(y)
         else:
            self._y_test.append(y)

      self._y_All = np.asarray(self._y_All)
      self._y_train = np.asarray(self._y_train)
      self._y_test = np.asarray(self._y_test)

   def y_categorical_dataset(self):
      """
         This function generates y (label)
         dataset to be used in a neural network
         classifier
      """

      for molecule in self._molecules:
         y = molecule.get_prop(self._propname) 

         """
         Convert y into a one-hot encoded binary vector.
         The lenght of the vector is equal to the number 
         of class_labels. All elements of the hot vector are
         zero but one element.
         
         Example: If we have four class_labels then 
                  hv = [0, 0, 0 ,1]
         """
         hv = hotvec(y.get_value(), self._class_labels)
         hv = np.fromiter(hv, dtype=np.int)

         self._y_All.append(hv)

         if molecule.is_intrain_set():
            self._y_train.append(hv)
         else:
            self._y_test.append(hv)

      self._y_All = np.asarray(self._y_All)
      self._y_train = np.asarray(self._y_train)
      self._y_test = np.asarray(self._y_test)

      if not self.check_all_labels_have_representative():
         sys.exit("At least one lable dose not have any representative in your dataset")

   def check_all_labels_have_representative(self):
      """
         This function checks to make sure all labels 
         have at least one representative in the dataset.

         If this is true, the sum over all the one-hot 
         binary vectors must be equal to the lenght of 
         the test and training sets. 
      """
      sum_test  = np.sum(self._y_test)
      sum_train = np.sum(self._y_train)
      
      if ((sum_test != len(self._y_test)) or 
         (sum_train != len(self._y_train))):
         return False

      return True

   def load_data(self)->bool:
      if self.gen_mols():
         self.train_test_split()
         self.X_generator()
         if self._class_labels:
            self.y_generator(categorical=True)
         else:
            self.y_generator(categorical=False)
         return True
      else:
         return False

   def x_train(self):
      return self._X_train

   def x_test(self):
      return self._X_test
   
   def y_train(self):
      return self._y_train

   def y_test(self):
      return self._y_test

   def get_molecules(self):
      return self._molecules

   def get_training_set(self):
      return self._training_set

   def get_test_set(self):
      return self._test_set

   def describe_data(self):
      print("\t\tTotal number of data points: %d" % len(self._molecules))
      print("\t\tNumber of data points in the training set: %s" % len(self._y_train))
      print("\t\tNumber of data points in the test set: %s" % len(self._y_test))

   def histogram_data(self, filename:str="data_histogram"):
      all_data = np.concatenate((self._y_train, self._y_test))
      histogram_plot(self._y_train, 
                     self._y_test, 
                     all_data,
                     self._propname, 
                     filename)
