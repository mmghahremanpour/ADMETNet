"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Utils.Imports import *
from admetnet.ML.Callbacks  import *

class DMEnumMeta(EnumMeta):
   def __contains__(cls, item):
      return item in cls.__members__

@unique
class FileFormat(Enum, metaclass=DMEnumMeta):

   NONE = 0

   csv  = 1
   json = 2
   sdf  = 3
   xyz  = 4
   pdb  = 5

   @classmethod
   def string_to_fileformat(cls, string):
      if string in FileFormat:
         return FileFormat[string]
      else:
         return cls.NONE

@unique
class DataSource(Enum, metaclass=DMEnumMeta):

   NONE = 0

   Experiment       = 1
   Theory           = 2
   Machine_Learning = 3

   @classmethod
   def string_to_datasource(cls, string):
      if string in DataSource:
         return DataSource[string]
      else:
         return cls.NONE

@unique
class JsonKeys(Enum, metaclass=DMEnumMeta):
   Discription = 1
   Units = 2
   Molecules = 3
   Properties = 4
   Molname = 5
   SMILES = 6
   Learning_Method = 7
   Learning_Type = 8

@unique
class LearningMethod(Enum, metaclass=DMEnumMeta):

   NONE = 0
   Random_Forest = 1
   Neural_Network = 2

   @classmethod
   def string_to_learningmethod(cls, string):
      if string in LearningMethod:
         return LearningMethod[string]
      else:
         return cls.NONE

@unique
class NetworkType(Enum, metaclass=DMEnumMeta):

   NONE = 0
   """Graph Convolutional Network """
   GCN = 1
   """Graph Attention Neural"""
   GAT = 2
   """Fully Connected Neural Network"""
   FCNN = 3

   @classmethod
   def string_to_networktype(cls, string):
      if string in NetworkType:
         return NetworkType[string]
      else:
         return cls.NONE

@unique
class LearningTask(Enum, metaclass=DMEnumMeta):
   
   NONE = 0
   Classification = 1
   Regression = 2

   @classmethod
   def string_to_learningtask(cls, string):
      if string in LearningTask:
         return LearningTask[string]
      else:
         return cls.NONE

@unique
class Featurizer(Enum, metaclass=DMEnumMeta):

   NONE = 0

   Morgan_Fingerprint = 1
   Atom_Pair_Fingerprint = 2
   Atom_Center_Fingerprint = 3 
   Topological_Torsion_Fingerprint = 4
   MolGraph = 5

   @classmethod
   def string_to_featurizer(cls, string):
      if string in Featurizer:
         return Featurizer[string]
      else:
         return cls.NONE


@unique
class Initializer(Enum, metaclass=DMEnumMeta):
   """
   Enum of distribution functions supported 
   by DeepModeller to generate initial values
   for weights and biases in each layer of 
   the neural network.
   """

   NONE = 0

   zeros = 1
   ones = 2
   random_uniform = 3
   random_normal  = 4
   glorot_uniform  = 5
   glorot_normal = 6
   he_uniform = 7
   he_normal = 8

   @classmethod
   def string_to_initializer(cls, string):
      if string in Initializer: 
         return Initializer[string]
      else:
         cls.NONE 

@unique
class LossFunction(Enum, metaclass=DMEnumMeta):

   NONE = 0

   """Regression losses """
   mean_squared_error = 1
   mean_absolute_error = 2
   mean_absolute_percentage_error = 3
   mean_squared_logarithmic_error = 4
   cosine_similarity = 5

   """Probabilistic losses for classification """
   binary_crossentropy = 6
   categorical_crossentropy = 7
   poisson = 8

   @classmethod
   def regression_loss(cls):
      return [cls.mean_squared_error, cls.mean_absolute_error,
              cls.mean_absolute_percentage_error, 
              cls.mean_squared_logarithmic_error,
              cosine_similarity]

   @classmethod
   def probabilistic_loss(cls):
      return [cls.binary_crossentropy, cls.categorical_crossentropy, cls.poisson]

   @classmethod
   def string_to_loss_function(cls, string):
      if string in LossFunction:
         return LossFunction[string]
      else:
         return cls.NONE

@unique
class Metric(Enum, metaclass=DMEnumMeta):
   """
   Enum class of Keras metrics supported 
   by DeepModeller

   Note:
   ----
   DeepModeller converts labels for classification
   modelling into a one-hot encoded binary vector. Thus, 
   sparse_categorical_crossentropy is not supported.
   """

   NONE = 0


   """Regression metrics """
   mean_squared_error = 1
   root_mean_squared_error = 2
   mean_absolute_error = 3
   mean_absolute_percentage_error = 4
   mean_squared_logarithmic_error = 5
   cosine_similarity = 6

   """Accuracy metrics """
   accuracy = 7
   binary_accuracy = 8
   categorical_accuracy = 9
   
   """Probabilistic metrics """
   binary_crossentropy = 10
   categorical_crossentropy = 11

   @classmethod
   def regression_metrics(cls):
      return [cls.mean_squared_error, cls.root_mean_squared_error,
              cls.mean_absolute_error, cls.mean_absolute_percentage_error,
              cls.mean_squared_logarithmic_error, cosine_similarity]

   @classmethod
   def accuracy_metrics(cls):
      return [cls.accuracy, cls.binary_accuracy, cls.categorical_accuracy]

   @classmethod
   def probabilistic_metrics(cls):
      return [cls.binary_crossentropy, cls.categorical_crossentropy] 

   @classmethod
   def binary_metrics(cls):
      """
      Returns a list of Keras metrics for two-class classification modelling
      """
      return [cls.binary_accuracy, cls.binary_crossentropy]

   @classmethod
   def categorical_metrics(cls):
      """
      Returns a list of Keras metrics for multi-class (more than two classes)
      classification modelling
      """
      return [cls.categorical_accuracy, cls.categorical_crossentropy]

   @classmethod
   def string_to_metric(cls, string):
      if string in Metric:
         return Metric[string]
      else:
         return cls.NONE

@unique
class ActivationFunction(Enum, metaclass=DMEnumMeta):

   NONE = 0

   elu  = 1 #Exponential Linear Unit
   selu = 2 #Scaled Exponential Linear Unit
   relu = 3 #REctified Linear Unit
   gelu = 4 #Gaussian Error Linear Unit
   tanh = 5
   sigmoid = 6
   hard_sigmoid = 7
   exponential = 8
   softmax = 9
   softplus = 10
   softsign = 11
   swish = 12
   linear = 13
   identity = 14 
   logistic = 15

   @classmethod
   def string_to_activation_function(cls, string):
      if string in ActivationFunction:
         return ActivationFunction[string]
      else:
         return cls.NONE   

@unique
class Regularizer(Enum, metaclass=DMEnumMeta):

   NONE = 0

   l1 = 1
   l2 = 2

   @classmethod
   def string_to_regularizer(cls, string):
      if string in Regularizer:
         return Regularizer[string]
      else:
         return cls.NONE   

   @classmethod
   def L1(cls, l1=0.01)->Callable:
      return tf.keras.regularizers.L1(l1=l1)

   @classmethod
   def L2(cls, l2=0.01)->Callable:
      return tf.keras.regularizers.L2(l2=l2)

@unique
class Optimizer(Enum, metaclass=DMEnumMeta):

   NONE = 0

   adam = 1
   sgd  = 2
   
   @classmethod
   def string_to_optimizer(cls, string):
      if string in Optimizer:
         return Optimizer[string]
      else:
         return cls.NONE

   @classmethod
   def Adam(cls, 
            learning_rate:float=1e-3, 
            beta1:float=0.9, 
            beta2:float=0.999, 
            epsilon:float=1e-8)->Callable:

      return tf.keras.optimizers.Adam(
               learning_rate=learning_rate,
               beta_1=beta1,
               beta_2=beta2,
               epsilon=epsilon)

   @classmethod
   def SGD(cls, learning_rate:float=1e-3)->Callable:
      return tf.keras.optimizers.SGD(learning_rate=learning_rate)

@unique
class Callback(Enum, metaclass=DMEnumMeta):

   NONE = 0

   logger      = 1
   plotter     = 2
   tensorboard = 3
   checkpoint  = 4
   realtimeplot = 5

   @classmethod
   def string_to_callback(cls, string):
      if string in Callback:
         return Callback[string]
      else:
         return cls.NONE

   @classmethod
   def Logger(cls,
              filename="training_curve.csv"):

      return DMLogger(filename=filename)

   
   @classmethod
   def Plotter(cls,
               filename="training_curve.xvg"):

      return GracePlotter(filename=filename)
  
   @classmethod
   def RealTimePlotter(cls):
      return RealTimePlot()

   @classmethod
   def TensorBoard(cls,
                   log_dir="tensorboard_logs"):

      return tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                            histogram_freq=1, 
                                            write_grads=True)

   @classmethod
   def CheckPoint(cls,
                  filepath="checkpoints_dir/checkpoint"):

      return tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True)
@unique
class MolecularProperties(Enum, metaclass=DMEnumMeta):
   Formalcharge = 1
   Dipole = 2
   Quadrupole = 3
   Polarizability = 4
   Enthapy_of_Formation = 5
   Entropy = 6
   Heat_Capacity = 7
   Cytotoxicity = 8
   Solubility = 9
   Permeability = 10
   LogP = 11
   LogS = 12
   CC50 = 13
   EC50 = 14
   IC50 = 15

@unique
class AtomicProperties(Enum, metaclass=DMEnumMeta):
   Charge = 1 
   Dipole = 2
   Quadrupole = 3
   Polarizability = 4
   Atomization_Energy = 5
   Electronegativity = 6
   Hardness = 7
   Ionization_Potential = 8
   Electron_Affinity = 9


