
from admetnet.Utils.Imports import *
from admetnet.Utils.Enums   import *
from admetnet.Utils.Status  import *

class DMParser:

   def __init__(self):

      self._db         = None
      self._db_file    = None
      self._db_params  = None
      self._fileformat = FileFormat.json
      self._dbsource   = DataSource.Experiment.name

      self._propname   = None
      self._featurizer = None
      self._explicit_hydrogen = None
      self._atom_degree  = None
      self._implicit_valence  = None
      self._ring_size = None
      self._atomic_hybridization = None
      self._atomic_number = None
      self._atom_descriptors = None
      self._edge_features = None
      self._node_features = None
      self._formal_charge = None
      self._number_of_hydrogens = None
      self._aromaticity = None
      self._chirality = None
      self._bond_order = None
      self._bond_conjugation = None
      self._morgan_fp_radius = None
      self._atompair_fp_length = None
      self._fingerprint_asBitVec = None
      self._fingerprint_nBits = None
      self._test_to_train_ratio = None
      self._class_labels = None
      self._bootstrap = None
      self._shuffle = None

      self._max_num_atom:int = 0

   def get_database(self):
      return self._db_file

   def set_database(self, database):
      self._db_file = database

   def get_database_paramfile(self):
      return self._db_params

   def set_database_paramfile(self, paramfile):
      self._db_params = paramfile

   def get_fileformat(self):
      return self._fileformat

   def set_fileformat(self, fileformat):
      self._fileformat = FileFormat.string_to_fileformat(fileformat)
      if self._fileformat == FileFormat.NONE:
         Status.UnsupportedFileFormat(fileformat)

   def get_property_name(self):
      return self._propname

   def get_graph_features(self):
      features = dict()
      features["ring_size"] = self._ring_size
      features["aromaticity"] = self._aromaticity
      features["atomic_hybridization"] = self._atomic_hybridization
      features["atomic_number"] = self._atomic_number
      features["atom_degree"] = self._atom_degree
      features["atom_descriptors"] = self._atom_descriptors
      features["implicit_valence"] = self._implicit_valence
      features["number_of_hydrogens"] = self._number_of_hydrogens
      features["formal_charge"] = self._formal_charge
      features["chirality"] = self._chirality
      features["bond_order"] = self._bond_order
      features["bond_conjugation"] = self._bond_conjugation
      return features

   def read_database(self):
      if self._fileformat == FileFormat.json:
         with open(self._db_file) as json_file:
            self._db = json.load(json_file)
      elif self._fileformat == FileFormat.csv:
         pass
   
   def read_database_params(self):
      with open(self._db_params) as json_file:

         params = json.load(json_file)
         params = params["Parameters"]

         if params.get("molecular_property") is not None:
            self._propname = params["molecular_property"]
         else:
            sys.exit("No molecular property is specified to extract from database")

         if params.get("class_labels") is not None:
            self._class_labels = params["class_labels"]

         if params.get("explicit_hydrogen") is not None:
            self._explicit_hydrogen = bool(params["explicit_hydrogen"] == "True")

         if params.get("ring_size") is not None:
            self._ring_size = bool(params["ring_size"] == "True")

         if params.get("atom_degree") is not None:
            self._atom_degree = bool(params["atom_degree"] == "True")

         if params.get("implicit_valence") is not None:
            self._implicit_valence = bool(params["implicit_valence"] == "True")

         if params.get("number_of_hydrogens") is not None:
            self._number_of_hydrogens = bool(params["number_of_hydrogens"] == "True")

         if params.get("atomic_hybridization") is not None:
            self._atomic_hybridization = bool(params["atomic_hybridization"] == "True")

         if params.get("atomic_number") is not None:
            self._atomic_number = bool(params["atomic_number"] == "True")

         if params.get("atom_descriptors") is not None:
            self._atom_descriptors = bool(params["atom_descriptors"] == "True")

         if params.get("edge_features") is not None:
            self._edge_features = bool(params["edge_features"] == "True")

         if params.get("node_features") is not None:
            self._node_features = bool(params["node_features"] == "True")

         if params.get("bond_order") is not None:
            self._bond_order = bool(params["bond_order"] == "True")

         if params.get("bond_conjugation") is not None:
            self._bond_conjugation = bool(params["bond_conjugation"] == "True")

         if params.get("chirality") is not None:
            self._chirality = bool(params["chirality"] == "True")

         if params.get("aromaticity") is not None:
            self._aromaticity = bool(params["aromaticity"] == "True")

         if params.get("formal_charge") is not None:
            self._formal_charge = bool(params["formal_charge"] == "True")

         if params.get("fingerprint_asBitVec") is not None:
            self._fingerprint_asBitVec = bool(params["fingerprint_asBitVec"] == "True")

         if params.get("fingerprint_nBits") is not None:
            self._fingerprint_nBits = params["fingerprint_nBits"]

         if params.get("morgan_fp_radius") is not None:
            self._morgan_fp_radius = params["morgan_fp_radius"]

         if params.get("atompair_fp_length") is not None:
            self._atompair_fp_length = params["atompair_fp_length"]

         if params.get("bootstrap") is not None:
            self._bootstrap = bool(params["bootstrap"] == "True")

         if params.get("shuffle") is not None:
            self._shuffle = bool(params["shuffle"] == "True")

         if params.get("test_to_train_ratio") is not None:
            self._test_to_train_ratio = params["test_to_train_ratio"]

            if self._test_to_train_ratio == 0.0:
               Status.NoTrainSet()
            elif self._test_to_train_ratio <= 0.1:
               Status.SmallTrainSet()
            elif self._test_to_train_ratio >= 0.9:
               Status.SmallTestSet()
            else:
               pass
      
         if params.get("featurizer") is not None:
            string = params["featurizer"]
            self._featurizer = Featurizer.string_to_featurizer(string)
            if self._featurizer == Featurizer.NONE:
               Status.UnsupportedHyperParameter(string)

         if params.get("data_source") is not None:
            string = params["data_source"]
            self._dbsource = DataSource.string_to_datasource(string)
            if self._dbsource == DataSource.NONE:
               Status.UnsupportedHyperParameter(string)

   def check_database(self)->bool:
      if JsonKeys.Molecules.name not in self._db.keys():
         Status.NoMolculeInData(self._db_file)
      for molecule in self._db[JsonKeys.Molecules.name]:
         if JsonKeys.SMILES.name not in molecule.keys():
            Status.NoSMILESInData(self._db_file)
         if not molecule[JsonKeys.Molname.name]:
            Status.NoIUPACInData(self._db_file)
         if not molecule[JsonKeys.Properties.name]:
            Status.NoRefPropertyInData(self._db_file)
      return True

