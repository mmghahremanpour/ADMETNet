
from admetnet.Utils.Imports import *
from admetnet.ML.Layers     import *
      
class FCNN(tf.keras.Model):
   """
      Class of Fully Connected Neural Network

      Attributes:
      ----------
      layer_sizes: A list of sizes of hidden layers. The length of the list
                   determines the number of hidden layers.

      w_initializer: Initializer distribution for weigths
      b_initializer: Initializer distribution for biases
      activation_fn: Activation function for the hidden layers. Activation 
                     function for the output layer depends on the learning type. 
                     It is linear for numerical regression and is softmax for classification. 

      regularizer:   Regularization function
      learning_task: Learning type: Rgression or Classification
      num_of_classes: Number of classes
   """

   def __init__(self, 
                dense_layers:List[int], 
                w_initializer=None,
                b_initializer=None,
                activation_fn=None,
                regularizer=None,
                learning_task=LearningTask.Regression,
                number_of_classes:int=None,
                verbose:bool=False,
                **kwargs):

      super(FCNN, self).__init__(**kwargs)

      self._learning_task = learning_task
      self._number_of_classes = number_of_classes
      self._w_initializer = w_initializer
      self._b_initializer = b_initializer
      self._activation_fn = activation_fn
      self._regularizer = regularizer
      self._verbose = verbose

      if not hasattr(dense_layers, "__iter__"):
         dense_layers = [dense_layers]

      if np.any(np.array(dense_layers) <= 0):
         raise ValueError("layer_sizes must be > 0, got %s." % dense_layers)

      self._dense_layers:List[int] = dense_layers
      self._layers:List = list()

      for i, size in enumerate(self._dense_layers):
         layer = DMDense(output_channels=size,
                         w_initializer=self._w_initializer,
                         b_initializer=self._b_initializer,
                         activation_fn=self._activation_fn,
                         activity_regularizer=self._regularizer,
                         w_regularizer=self._regularizer,
                         b_regularizer=self._regularizer,
                         verbose=self._verbose,
                         name="DMDense_Hidden_{}".format(i)
                        )
      
         self._layers.append(layer)      

      if self._learning_task == LearningTask.Regression:
         layer = DMDense(output_channels=1, 
                         w_initializer=self._w_initializer,
                         b_initializer=self._b_initializer,
                         activation_fn=ActivationFunction.linear.name,
                         activity_regularizer=self._regularizer,
                         w_regularizer=self._regularizer,
                         b_regularizer=self._regularizer,
                         verbose=self._verbose,
                         name="DMDense_OUT"
                        )

         self._layers.append(layer)

      elif self._learning_task == LearningTask.Classification:
         if self._num_of_classes:
            layer = DMDense(output_channels=self._num_of_classes, 
                            w_initializer=self._w_initializer,
                            b_initializer=self._b_initializer,
                            activation_fn=ActivationFunction.softmax.name,
                            activity_regularizer=self._regularizer,
                            w_regularizer=self._regularizer,
                            b_regularizer=self._regularizer,
                            verbose=self._verbose,
                            name="DMDense_OUT"
                           )

            self._layers.append(layer)

         else:
            pass
      else:
         pass

   def call(self, inputs):
      outputs = inputs

      for layer in self._layers:
         outputs = layer(outputs)

      return outputs

class GCNN:
   """
   Graph Convolution Neural Network
   """
   def __init__(self, 
                num_node_attr,
                num_edge_attr,
                conv_layer_sizes:List[int]=[32, 32],
                dense_layer_sizes:List[int]=[16, 16], 
                add_bias:bool=True,
                w_initializer:str=Initializer.glorot_normal.name,
                b_initializer:str=Initializer.zeros.name,
                activation_fn:str=ActivationFunction.relu.name,
                regularizer:str=None,
                learning_task:str=LearningTask.Regression,
                number_of_classes:int=None,
                verbose:bool=False):

      if not hasattr(dense_layer_sizes, "__iter__"):
         dense_layer_sizes = [dense_layer_sizes]

      if np.any(np.array(dense_layer_sizes) <= 0):
         raise ValueError("dense_layer_sizes must be > 0, got %s." % dense_layer_sizes)

      if not hasattr(conv_layer_sizes, "__iter__"):
         conv_layer_sizes = [conv_layer_sizes]

      if np.any(np.array(conv_layer_sizes) <= 0):
         raise ValueError("conv_layer_sizes must be > 0, got %s." % conv_layer_sizes)

      self._conv_layer_sizes:List[int] = conv_layer_sizes
      self._dense_layer_sizes:List[int] = dense_layer_sizes

      self._learning_task = learning_task
      self._num_of_classes = number_of_classes
      self._verbose = verbose

      # Input layers
      self.node_input = tf.keras.Input((None, num_node_attr))
      self.adj_input = tf.keras.Input((None, None))

      x = [self.node_input, self.adj_input]

      # Convolution layers
      for i, size in enumerate(self._conv_layer_sizes):
         x = DMGCN(output_channels=size,
                   add_bias=True,
                   w_initializer=w_initializer,
                   b_initializer=b_initializer,
                   activation_fn=activation_fn,
                   activity_regularizer=regularizer,
                   w_regularizer=regularizer,
                   b_regularizer=regularizer,
                   verbose=self._verbose,
                   name="DMGCN_{}".format(i)
                  )(x)

      # Reduce to graph features
      x = DMGReduce(name="DMGReduce_1")(x)

      # Hidden dense layers
      for i, size in enumerate(self._dense_layer_sizes):
         x = DMDense(output_channels=size, 
                     w_initializer=w_initializer,
                     b_initializer=b_initializer,
                     activation_fn=activation_fn,
                     activity_regularizer=regularizer,
                     w_regularizer=regularizer,
                     b_regularizer=regularizer,
                     verbose=self._verbose,
                     name="DMDense_Hidden_{}".format(i)
                    )(x)

      
      # Output layer
      if self._learning_task == LearningTask.Regression:
         x = DMDense(output_channels=1, 
                     w_initializer=w_initializer,
                     b_initializer=b_initializer,
                     activation_fn=ActivationFunction.linear.name,
                     activity_regularizer=regularizer,
                     w_regularizer=regularizer,
                     b_regularizer=regularizer,
                     verbose=self._verbose,
                     name="DMDense_OUT"
                    )(x)

      elif self._learning_task == LearningTask.Classification:
         x = DMDense(output_channels=self._num_of_classes, 
                     w_initializer=w_initializer,
                     b_initializer=b_initializer,
                     activation_fn=ActivationFunction.softmax.name,
                     activity_regularizer=regularizer,
                     w_regularizer=regularizer,
                     b_regularizer=regularizer,
                     verbose=self._verbose,
                     name="DMDense_OUT"
                    )(x)
      else:
         pass

      self._model = tf.keras.Model(inputs=(self.node_input, self.adj_input), outputs=x)
       
   def model(self):
      return self._model
            
class GANN:
   """
   Graph Attention Neural Network
   """
   def __init__(self, 
                num_node_attr,
                atten_heads=1,
                concat_atten_heads=False,
                atten_layer_sizes:List[int]=[32, 32],
                dense_layer_sizes:List[int]=[16, 16],
                add_bias:bool=True,
                dropout_rate=0.01,
                w_initializer:str=Initializer.glorot_uniform.name,
                b_initializer:str=Initializer.zeros.name,
                activation_fn=ActivationFunction.elu.name,
                regularizer:str=None,
                learning_task:str=LearningTask.Regression,
                number_of_classes:int=None,
                verbose:bool=False):

      if not hasattr(dense_layer_sizes, "__iter__"):
         dense_layer_sizes = [dense_layer_sizes]

      if np.any(np.array(dense_layer_sizes) <= 0):
         raise ValueError("dense_layer_sizes must be > 0, got %s." % dense_layer_sizes)

      if not hasattr(atten_layer_sizes, "__iter__"):
         atten_layer_sizes = [atten_layer_sizes]

      if np.any(np.array(atten_layer_sizes) <= 0):
         raise ValueError("atten_layer_sizes must be > 0, got %s." % atten_layer_sizes)

      self._atten_layer_sizes:List[int] = atten_layer_sizes
      self._dense_layer_sizes:List[int] = dense_layer_sizes

      self._learning_task = learning_task
      self._num_of_classes = number_of_classes
      self._verbose = verbose

      # Input layers
      self.node_input = tf.keras.Input((None, num_node_attr), name="Feature_Matrix")
      self.adj_input = tf.keras.Input((None, None), name="Adjacency_Matrix")

      x = [self.node_input, self.adj_input]

      # Attention layers
      for i, size in enumerate(self._atten_layer_sizes):
         x = DMGAttention(output_channels=size,
                          atten_heads=atten_heads,
                          concat_atten_heads=concat_atten_heads,
                          add_bias=add_bias,
                          dropout_rate=dropout_rate,
                          w_initializer=w_initializer,
                          b_initializer=b_initializer,
                          activation_fn=activation_fn,
                          activity_regularizer=regularizer,
                          w_regularizer=regularizer,
                          b_regularizer=regularizer,
                          verbose=verbose,
                          name="DMGAttention_{}".format(i)
                          )(x)


      # Reduce to graph features
      x = DMGReduce(name="DMGReduce_1")(x)

      # Hidden dense layers
      for i, size in enumerate(self._dense_layer_sizes):
         x = DMDense(output_channels=size, 
                     w_initializer=w_initializer,
                     b_initializer=b_initializer,
                     activation_fn=activation_fn,
                     activity_regularizer=regularizer,
                     w_regularizer=regularizer,
                     b_regularizer=regularizer,
                     verbose=self._verbose,
                     name="DMDense_Hidden_{}".format(i)
                    )(x)

      # Output layer
      if self._learning_task == LearningTask.Regression:
         x = DMDense(output_channels=1, 
                     w_initializer=w_initializer,
                     b_initializer=b_initializer,
                     activation_fn=ActivationFunction.linear.name,
                     activity_regularizer=regularizer,
                     w_regularizer=regularizer,
                     b_regularizer=regularizer,
                     verbose=self._verbose,
                     name="DMDense_OUT"
                    )(x)

      elif self._learning_task == LearningTask.Classification:
         x = DMDense(output_channels=self._num_of_classes, 
                     w_initializer=w_initializer,
                     b_initializer=b_initializer,
                     activation_fn=ActivationFunction.softmax.name,
                     activity_regularizer=regularizer,
                     w_regularizer=regularizer,
                     b_regularizer=regularizer,
                     verbose=self._verbose,
                     name="DMDense_OUT"
                    )(x)
      else:
         pass

      self._model = tf.keras.Model(inputs=(self.node_input, self.adj_input), outputs=x)
       
   def model(self):
      return self._model


