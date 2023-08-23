
from admetnet.Data.DataLoader import *
from admetnet.Utils.Enums import *

class HyperParameters:
   def __init__(self):
      self._n_estimators = None
      self._criterion    = None  
      self._max_depth    = None
      self._min_samples_split = None
      self._min_samples_leaf  = None
      self._min_weight_fraction_leaf = None
      self._max_features = None
      self._max_leaf_nodes = None
      self._min_impurity_decrease = None
      self._min_impurity_split = None
      self._bootstrap = None
      self._n_jobs = None
      self._random_state = None
      self._verbose_1  = None
      self._verbose_2  = None
      self._warm_start  = None
      self._max_samples = None
      self._oob_score = None
      self._ccp_alpha = None


      self._dense_layers = None
      self._graph_convolutional_layers = None
      self._graph_attention_layers = None
      self._graph_attention_heads = None
      self._network_type = None
      self._activation_fn = None  
      self._loss_fn = None
      self._callbacks = None
      self._metric = None
      self._regularizer = None
      self._penalty_rate = None
      self._optimizer = None
      self._alpha = None
      self._batch_size  = None
      self._learning_rate = None
      self._learning_rate_init = None
      self._w_initializer = None
      self._b_initializer = None
      self._power_t = None
      self._epochs = None
      self._shuffle = None
      self._tolerance = None
      self._momentum = None
      self._nesterovs_momentum = None
      self._early_stopping = None
      self._validation_fraction = None
      self._beta1 = None
      self._beta2 = None
      self._epsilon = None
      self._n_iter_no_change = None
      self._max_fun = None

class BModel(HyperParameters, DataLoader):
   """
      Abstract Base Model Class for all Model Subclasses

      This class should be only used 
      for implementing subclasses 

   """
   def __init__(self, 
                paramfile:str=None, 
                datafile:str=None, 
                modelfile:str=None,
                weightsfile:str=None,
                learning_method:str = LearningMethod.Neural_Network.name, 
                learning_task = LearningTask.Regression.name):

      HyperParameters.__init__(self)
      DataLoader.__init__(self)
      
      self._params = None
      self._cv = None
      self._paramfile:str = paramfile
      self._datafile:str  = datafile
      self._modelfile:str = modelfile
      self._weightsfile:str = weightsfile   
      self._learning_method:str = learning_method
      self._learning_task:str = learning_task
      self._num_graph_node_features = None
      self._num_graph_edge_features = None

      self.model_is_built = False
      self.model_is_compiled = False

   @property
   def graph_node_features(self):
      return self._num_graph_node_features

   @graph_node_features.setter
   def graph_node_features(self, n):
      self._num_graph_node_features = n

   @property
   def graph_edge_features(self):
      return self._num_graph_edge_features

   @graph_edge_features.setter
   def graph_edge_features(self, n):
      self._num_graph_edge_features = n

   @property
   def learning_method(self):
      return self._learning_method

   @learning_method.setter
   def learning_method(self, learning_method):
      self._learning_method = LearningMethod.string_to_learningmethod(learning_method)
      if self._learning_method == LearningMethod.NONE:
         Status.UnsupportedHyperParameter(learning_method)

   @property
   def learning_task(self):
      return self._learning_task

   @learning_task.setter
   def learning_task(self, learning_task):
      self._learning_task = LearningTask.string_to_learningtask(learning_task)
      if self._learning_task == LearningTask.NONE:
         Status.UnsupportedHyperParameter(learning_task)

   @property
   def network_type(self):
      return self._network_type

   @network_type.setter
   def network_type(self, network_type):
      self._network_type = NetworkType.string_to_networktype(network_type)
      if self._network_type == NetworkType.NONE:
         Status.UnsupportedHyperParameter(string)

   def get_modelfile(self):
      """
      Returns the filename for saving/loading a trained model
      """
      return self._modelfile

   def set_modelfile(self, filename):
      """
      Set the filename for saving/loading a trained model
      """
      self._modelfile = filename

   def get_weightsfile(self):
      """
      Returns the filename for saving/loading the weights 
      of a trained model.
      """
      return self._weightsfile

   def set_weightsfile(self, weightsfile):
      """
      Sets the filename for saving/loading the weights 
      of a trained model.
      """
      self._weightsfile = weightsfile

   def get_network_paramfile(self):
      """
      Returns the hyperparameter filename
      """
      return self._paramfile

   def set_network_paramfile(self, paramfile):
      """
      Set the hyperparameter filename
      """
      self._paramfile = paramfile

   def read_params(self, learning_method):
      """
      function to read the hyper parameters of 
      a model from a .json file
      """
      with open(self._paramfile) as json_file:
         params = json.load(json_file)
         self._params =  params[JsonKeys.Learning_Method.name][learning_method]
   
   def set_nn_params(self):
      """
      Set parameters for a Neural Network model
      """

      if self._params.get("dense_layers") is not None:
         self._dense_layers = self._params["dense_layers"]

      if self._params.get("graph_convolutional_layers") is not None:
         self._graph_convolutional_layers = self._params["graph_convolutional_layers"]

      if self._params.get("graph_attention_layers") is not None:
         self._graph_attention_layers = self._params["graph_attention_layers"]

      if self._params.get("graph_attention_heads") is not None:
         self._graph_attention_heads = self._params["graph_attention_heads"]

      if self._params.get("activation_fn") is not None:
         string = self._params["activation_fn"]  
         self._activation_fn = ActivationFunction.string_to_activation_function(string)
         if self._activation_fn == ActivationFunction.NONE:
             Status.UnsupportedHyperParameter(string)

      if self._params.get("callbacks") is not None:
         callbacks = []
         for callback in self._params["callbacks"]:
            callback = Callback.string_to_callback(callback)
            if callback == Callback.NONE:
               Status.UnsupportedHyperParameter(callback)
            else:
               callbacks.append(callback)
         self._callbacks = callbacks

      if self._params.get("network_type") is not None:
         string = self._params["network_type"]  
         self._network_type = NetworkType.string_to_networktype(string)
         if self._network_type == NetworkType.NONE:
             Status.UnsupportedHyperParameter(string)

         if (self._network_type == NetworkType.GCNN and 
             (self._graph_convolutional_layers == None)):
             Status.ZeroConvLayer()

         if (self._network_type == NetworkType.GANN and 
             (self._graph_attention_layers == None)):
             Status.ZeroAttentionLayer()

         if (self._network_type == NetworkType.GANN and 
             (self._graph_attention_heads == 0 or self._graph_attention_heads == None)):
             Status.ZeroAttentionHead()

      if self._params.get("loss_fn") is not None:
         string = self._params["loss_fn"]  
         self._loss_fn = LossFunction.string_to_loss_function(string)
         if self._loss_fn == LossFunction.NONE:
             Status.UnsupportedHyperParameter(string)  

      if self._params.get("w_initializer") is not None:
         string = self._params["w_initializer"]  
         self._w_initializer = Initializer.string_to_initializer(string) 
         if self._w_initializer == Initializer.NONE:
             Status.UnsupportedHyperParameter(string) 

      if self._params.get("b_initializer") is not None:
         string = self._params["b_initializer"]  
         self._b_initializer = Initializer.string_to_initializer(string)  
         if self._b_initializer == Initializer.NONE:
             Status.UnsupportedHyperParameter(string)

      if self._params.get("metric") is not None:
         string = self._params["metric"] 
         self._metric = Metric.string_to_metric(string)
         if self._metric == Metric.NONE:
             Status.UnsupportedHyperParameter(string)

      if self._params.get("optimizer") is not None: 
         string = self._params["optimizer"]
         self._optimizer = Optimizer.string_to_optimizer(string)
         if self._optimizer == Optimizer.NONE:
            Status.UnsupportedHyperParameter(string)
      
      if self._params.get("regularizer") is not None: 
         string = self._params["regularizer"]
         self._regularizer = Regularizer.string_to_regularizer(string)
         if self._regularizer == Regularizer.NONE:
            self._regularizer = None
            #Status.UnsupportedHyperParameter(string)

      if self._params.get("learning_task") is not None:
         string = self._params["learning_task"]
         self._learning_task = LearningTask.string_to_learningtask(string)
         if self._learning_task == LearningTask.NONE:
            Status.UnsupportedHyperParameter(string)

      if self._params.get("Learning_Method") is not None:
         string = self._params["Learning_Method"]
         self._learning_method = LearningMethod.string_to_learningmethod(string)
         if self._learning_method == LearningMethod.NONE:
            Status.UnsupportedHyperParameter(string)

      if self._params.get("alpha") is not None:
         self._alpha = float(self._params["alpha"])
      else:
         self._alpha = 0.0

      if self._params.get("penalty_rate") is not None:
         self._penalty_rate = float(self._params["penalty_rate"])
      else:
         self._penalty_rate = 0.0

      if self._params.get("batch_size") is not None:
         self._batch_size = self._params["batch_size"]

      if self._params.get("learning_rate") is not None:
         self._learning_rate = str(self._params["learning_rate"])

      if self._params.get("learning_rate_init") is not None:
         self._learning_rate_init = float(self._params["learning_rate_init"])
      else:
         self._learning_rate_init = 1e-3

      if self._params.get("power_t") is not None:
         self._power_t = float(self._params["power_t"])

      if self._params.get("epochs") is not None:
         self._epochs = int(self._params["epochs"])

      if self._params.get("shuffle") is not None:
         self._shuffle = bool(self._params["shuffle"] == "True")

      if self._params.get("random_state") is not None:
         self._random_state = int(self._params["random_state"])

      if self._params.get("tolerance") is not None:
         self._tolerance = float(self._params["tolerance"])

      if self._params.get("verbose_1") is not None:
         self._verbose_1  = bool(self._params["verbose_1"] == "True")

      if self._params.get("verbose_2") is not None:
         self._verbose_2  = self._params["verbose_2"]

      if self._params.get("warm_start") is not None:
         self._warm_start  = bool(self._params["warm_start"] == "True")

      if self._params.get("momentum") is not None:
         self._momentum = float(self._params["momentum"])

      if self._params.get("nesterovs_momentum") is not None:
         self._nesterovs_momentum = bool(self._params["nesterovs_momentum"] == "True")

      if self._params.get("early_stopping") is not None:
         self._early_stopping = bool(self._params["early_stopping"] == "True")
 
      if self._params.get("validation_fraction") is not None:
         self._validation_fraction = float(self._params["validation_fraction"])

      if self._params.get("beta_1") is not None:
         self._beta1 = float(self._params["beta_1"])
 
      if self._params.get("beta_2") is not None:
         self._beta2 = float(self._params["beta_2"])

      if self._params.get("epsilon") is not None:
         self._epsilon = float(self._params["epsilon"])

      if self._params.get("n_iter_no_change") is not None:
         self._n_iter_no_change = int(self._params["n_iter_no_change"])

      if self._params.get("max_fun") is not None:
         self._max_fun = int(self._params["max_fun"])

   def load_params(self) -> None:
      """
      function for loading hyperparameters

      Subclass should implement this.
      """
      raise NotImplementedError

   def check_params(self) -> None:
      """
      function for checking hyperparameters

      Subclass should implement this.
      """
      raise NotImplementedError

   def update_parameter(self, name, value):
      """
      function for chaning the value of a 
      hyperparameter

      Subclass should implement this.
      """
      raise NotImplementedError

   def make_model(self) -> None:
      """ 
      function for making a model 

      Subclass should implement this.
      """
      raise NotImplementedError

   def compile_model(self) -> None:
      """ 
      compile a nueral network model 

      Subclass should implement this.
      """
      raise NotImplementedError

   def save_model(self, overwrite=True) -> None:
      """ 
      function for saving a trained model 
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def summerize_model(self, overwrite=True) -> None:
      """ 
      summerize model 
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def save_weights(self, overwrite=True) -> None:
      """ 
      function for saving the weights of a trained model 
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def plot_model(self) -> None:
      """ 
      function to draw a picture of the layers and their shapes 
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def load_model(self) -> None:
      """ 
      function for loading a trained model 
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def load_weights(self) -> None:
      """ 
      function for loading the weights of a trained model 
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def train_model(self) -> None:
      """ 
      function for training a  model 
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def evaluate_model(self) -> None:
      """ 
      function for evaluating a model on the test set 
      
      Subclass should implement this.
      """
      raise NotImplementedError


   def predict_on_test(self) -> None:
      """ 
      function for making predicitons on the
      test set, this is for regression analysis
      between y and y_pred.  
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def predict_on_train(self) -> None:
      """ 
      function for making predicitons on the
      training set, this is for regression analysis
      between y and y_pred.  
      
      Subclass should implement this.
      """
      raise NotImplementedError

   def predict(self, x) -> None:
      """
      function for making prediction on the 
      input x.

      Subclass should implement this.
      """
      raise NotImplementedError

   def cross_validator(self) -> None:
      """
      function for cross validating a model
      using repeated K fold algorithm.

      Subclass should implement this.
      """
      raise NotImplementedError

   def evaluate_hyper_parameters(self) -> None:
      """
      function for evaluating hyperparameters 
      for a model, for example, different number 
      of hidden layers.

      Subclass should implement this.
      """
      raise NotImplementedError

