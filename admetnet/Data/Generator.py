"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""


from admetnet.Utils.Imports import *
from admetnet.Utils.Lambdas import *


class Generator:
   
   def __init__(self, 
                features=None,
                maxatom=None,
                edge_attr=None,
                n_node_attr=None,
                propname=None,
                labels=None):

      self.features    = features
      self.maxatom     = maxatom
      self.edgeattr    = edge_attr
      self.n_node_attr = n_node_attr
      self.propname    = propname
      self.labels      = labels

   def set_n_node_attr(self, n_node_attr):
      self.n_node_attr = n_node_attr

   def get_n_node_attr(self):
      return self.n_node_attr

   def set_features(self, features):
      self.features = features

   def get_features(self):
      return self.features

   def set_maxatom(self, maxatom):
      self.maxatom = maxatom

   def get_maxatom(self):
      return self.maxatom

   def set_edgeattr(self, edgeattr):
      self.edgeattr = edgeattr

   def get_edgeattr(self):
      return self.edgeattr

   def set_propname(self, propname):
      self.propname = propname

   def set_labels(self, labels):
      self.labels = labels

   def get_labels(self):
      return self.labels

   def regression(self, molset=None):

      if molset == None:
         sys.exit("molset is empty")

      dataset = None

      def generator():
         for molecule in molset:
            molecule.molgraph(features=self.features)
            X = molecule.graph_node_attr_adj_matrix(max_num_node=self.maxatom, 
                                                    add_edge_attr=self.edgeattr)
            y = molecule.get_prop(self.propname)
            y = float(y.get_value())
            yield X, y
      
      dataset = tf.data.Dataset.from_generator(generator, 
                                               output_types=((tf.float32, tf.float32), tf.float32), 
                                               output_shapes=((tf.TensorShape([None, self.n_node_attr]), 
                                               tf.TensorShape([None, None])), 
                                               tf.TensorShape([])))

      return dataset

   def classification(self, molset=None):

      def generator():
         for molecule in molset:
            molecule.molgraph(features=self.features)
            X = molecule.graph_node_attr_adj_matrix(max_num_node=self.maxatom, 
                                                    add_edge_attr=self.edgeattr)
            y = molecule.get_prop(self.propname)
            labels = hotvec(y.get_value(), self.labels)
            labels = np.fromiter(labels, dtype=np.int)
            yield X, labels
            
      dataset = tf.data.Dataset.from_generator(generator, 
                                               output_types=((tf.float32, tf.float32), (tf.int32)), 
                                               output_shapes=((tf.TensorShape([None, self.n_node_attr]), 
                                               tf.TensorShape([None, None])), 
                                               tf.TensorShape([4])))
      return dataset
