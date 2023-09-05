"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.ML.BaseModel  import *
from admetnet.ML.Models     import *
from admetnet.ML.Callbacks  import *
from admetnet.Utils.Enums   import *
from admetnet.Utils.Imports import *
from admetnet.Utils.Status  import *


class Networks(BModel):

   def __init__(self):
      BModel.__init__(self)
      self._model = None

   def load_params(self):
      self.read_params(LearningMethod.Neural_Network.name)
      self.set_nn_params()
         
   def update_parameter(self, name, value):
      self._params[name] = value
      self.set_nn_params()

   def make_model(self):

      self.load_params()

      number_of_classes = None
      if self.learning_task == LearningTask.Classification:
         _, number_of_classes = self._y_train.shape

      if self._regularizer == Regularizer.l1:
         self._regularizer = Regularizer.L1(l1=self._penalty_rate)
      elif self._regularizer == Regularizer.l2:
         self._regularizer = Regularizer.L2(l2=self._penalty_rate)
      else:
         self._regularizer = None

      if self._network_type == NetworkType.FCNN:
         self._model = FCNN(dense_layers=self._dense_layers,
                            w_initializer=self._w_initializer.name,
                            b_initializer=self._b_initializer.name,
                            activation_fn=self._activation_fn.name,
                            regularizer=self._regularizer,
                            learning_task=self._learning_task,
                            number_of_classes=number_of_classes)
         self.model_is_built = True

      else:
         if self._network_type == NetworkType.GCN:
            self._model = GCNN(num_node_attr=self._num_graph_node_features,
                               num_edge_attr=self._num_graph_edge_features,
                               conv_layer_sizes=self._graph_convolutional_layers,
                               dense_layer_sizes=self._dense_layers,
                               add_bias=True,
                               w_initializer=self._w_initializer.name,
                               b_initializer=self._b_initializer.name,
                               activation_fn=self._activation_fn.name,
                               regularizer=self._regularizer,
                               learning_task=self._learning_task,
                               number_of_classes=number_of_classes,
                               verbose=self._verbose_1
                               ).model()
            self.model_is_built = True

         elif self._network_type == NetworkType.GAT:
            self._model = GANN(num_node_attr=self._num_graph_node_features,
                               atten_heads=self._graph_attention_heads,
                               concat_atten_heads=False,
                               atten_layer_sizes=self._graph_attention_layers,
                               dense_layer_sizes=self._dense_layers,
                               add_bias=True,
                               dropout_rate=0.01,
                               w_initializer=self._w_initializer.name,
                               b_initializer=self._b_initializer.name,
                               activation_fn=self._activation_fn.name,
                               regularizer=self._regularizer,
                               learning_task=self._learning_task,
                               number_of_classes=number_of_classes,
                               verbose=self._verbose_1
                               ).model()
            self.model_is_built = True

         else:
            pass


   def compile_model(self):

      if self.model_is_built:
         optimizer = None
         if self._optimizer == Optimizer.adam:
            optimizer = Optimizer.Adam(learning_rate=self._learning_rate_init,
                                       beta1=self._beta1,
                                       beta2=self._beta2,
                                       epsilon=self._epsilon)
         elif self._optimizer == Optimizer.sgd:
            optimizer = Optimizer.SGD(learning_rate=self._learning_rate_init)
         else:
            Status.UnsupportedOptimizer(self._optimizer.name) 

         self._model.compile(optimizer=optimizer, 
                             loss=self._loss_fn.name,
                             metrics=[self._metric.name])

         self.model_is_compiled = True
      else:
         sys.exit("Model is not built, so it cannot be compiled.")

   def preprocess(self):

      features = self.get_graph_features()
      num_node_features, num_edge_features  = DMMol().graph_features(features=features)
      self.graph_node_features = num_node_features
      self.graph_edge_features = num_edge_features

      self.make_model()
      self.compile_model()


   def train_model(self):

      if self.model_is_compiled:
         callbacks = []
         if self._callbacks:
            for callback in self._callbacks:
               if callback == Callback.logger:
                  callback = Callback.Logger()
               elif callback == Callback.plotter:
                  callback = Callback.Plotter()
               elif callback == Callback.tensorboard:
                  callback = Callback.TensorBoard()
               elif callback == Callback.checkpoint:
                  callback = Callback.CheckPoint()
               elif callback == Callback.realtimeplot:
                  callback = Callback.RealTimePlotter()
               callbacks.append(callback)

         if self._network_type == NetworkType.FCNN:
            self._model.fit(self._X_train, 
                            self._y_train, 
                            batch_size=self._batch_size, 
                            epochs=self._epochs, 
                            verbose=self._verbose_2, 
                            validation_data=(self._X_test, self._y_test), 
                            callbacks=callbacks)
         else:
            self._model.fit(self._train_data.batch(self._batch_size), 
                            validation_data=self._val_data.batch(self._batch_size),  
                            epochs=self._epochs, 
                            verbose=self._verbose_2,
                            callbacks=callbacks)
      else:
         sys.exit("Model is not compiled, so it cannot be trained.")

   def predict(self, x=None):
      if x is not None:
         if self._network_type == NetworkType.FCNN:
            y_pred = self._model.predict(x)
         else:
            y_true = []
            for data in list(x.as_numpy_iterator()):
               X, y = data
               y_true.append(y)

            y_true = np.array(y_true)
            y_pred = self._model.predict(x.batch(1))

            if y_pred.ndim == 3:
               y_pred = np.squeeze(y_pred, axis=2)
         return y_pred
      else:
         return None

   def predict_on_test(self):
      if self._network_type == NetworkType.FCNN:
         y_true = self._y_test
         y_pred = self._model.predict(self._X_test)
      else:
         y_true = []
         for data in list(self._test_data.as_numpy_iterator()):
            X, y = data
            y_true.append(y)
         y_true = np.array(y_true)
         y_pred = self._model.predict(self._test_data.batch(self._batch_size))

         if y_pred.ndim == 3:
            y_pred = np.squeeze(y_pred, axis=2)

      return y_true, y_pred

   def predict_on_train(self):
      if self._network_type == NetworkType.FCNN:
         y_true = self._y_train
         y_pred = self._model.predict(self._X_train)
      else:
         y_true = []
         for data in list(self._train_data.as_numpy_iterator()):
            X, y = data
            y_true.append(y)
         y_true = np.array(y_true)
         y_pred = self._model.predict(self._train_data.batch(self._batch_size))

         if y_pred.ndim == 3:
            y_pred = np.squeeze(y_pred, axis=2)

      return y_true, y_pred

   def predict_on_dataset(self):
      if self._network_type == NetworkType.FCNN:
         y_true = self._y_All
         y_pred = self._model.predict(self._X_All)
      else:
         y_true = []
         for data in list(self._dataset.as_numpy_iterator()):
            X, y = data
            y_true.append(y)
         y_true = np.array(y_true)
         y_pred = self._model.predict(self._dataset.batch(self._batch_size))

         if y_pred.ndim == 3:
            y_pred = np.squeeze(y_pred, axis=2)

      return y_true, y_pred

   def evaluate(self):
      self._model.evaluate(self._X_test, 
                           self._y_test, 
                           batch_size=self._batch_size, 
                           verbose=self._verbose_1)      

   def save_model(self, overwrite=True):
      if not overwrite:
         overwrite = ask_user_to_overwrite(self._modelfile)
         if not overwrite:
            return
      tf.keras.models.save_model(self._model, self._modelfile, include_optimizer=False)

   def to_json(self):
      config = self._model.to_json()
      with open(self._modelfile, 'w') as outfile:
         json.dump(config, outfile)

   def get_weights(self):
      weigths = defaultdict(dict)
      for layer in self._model.layers:
         w    = layer.get_weights()
         name = layer.name
         if len(w) == 1:
            weigths[name]["kernel"] = w[0]
         elif len(w) == 2:
            weigths[name]["kernel"] = w[0]
            weigths[name]["bias"]   = w[1]
      return weigths

   def save_weights(self):
      self._model.save_weights(self._weightsfile, save_format='h5')

   def get_layers(self):
      return self._model.layers

   def plot_model(self):
      tf.keras.utils.plot_model(self._model, to_file='model.png', show_shapes=True,)

   def load_model(self):
      self._model = tf.keras.models.load_model(self._modelfile, 
                                               custom_objects = {'HeUniform': tf.keras.initializers.he_uniform(),
                                                                 'GlorotNormal' : tf.keras.initializers.glorot_normal(),
                                                                 'Zeros': tf.keras.initializers.zeros(),
                                                                 'L1L2': tf.keras.regularizers.l2()},
                                                                 compile=False)

   def load_model_from_json(self):
      with open(self._modelfile) as json_file:
         config = json.load(json_file)
         self._model = tf.keras.models.model_from_json(config,
                                                       custom_objects = {'HeUniform': tf.keras.initializers.he_uniform(),
                                                       'GlorotNormal' : tf.keras.initializers.glorot_normal(),
                                                       'Zeros': tf.keras.initializers.zeros(),
                                                       'L1L2': tf.keras.regularizers.l2()})

   def load_weights(self):
      self._model.load_weights(self._weightsfile, by_name=True)

   def summerize_model(self):
      if self._model:
         self._model.summary()
