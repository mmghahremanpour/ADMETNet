"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Utils.Imports import *
from admetnet.Utils.Enums   import *
from admetnet.Utils.Status  import *

@tf.keras.utils.register_keras_serializable()
class DMDense(tf.keras.layers.Layer):
   """
   ADMETNet Dense Layer
   """
   def __init__(self, 
                output_channels:int=64,
                add_bias:bool=True,
                w_initializer:str=Initializer.glorot_normal.name,
                b_initializer:str=Initializer.zeros.name,
                activation_fn:str=None,
                activity_regularizer:str=None,
                w_regularizer:str=None,
                b_regularizer:str=None,
                verbose:bool=False,
                name=None,
                **kwargs):


      self.output_channels = int(output_channels)

      self.w_regularizer        = w_regularizer
      self.b_regularizer        = b_regularizer
      self.activity_regularizer = activity_regularizer

      self.w_initializer = tf.keras.initializers.get(w_initializer)
      self.b_initializer = tf.keras.initializers.get(b_initializer)
      self.activation_fn = tf.keras.activations.get(activation_fn)

      self.add_bias   = add_bias
      self.verbose    = verbose
      self.layer_name = name

      super(DMDense, self).__init__(name=self.layer_name,
                                    activity_regularizer=self.activity_regularizer, 
                                    **kwargs)
   def build(self, input_shape):
      input_shape = tf.TensorShape(input_shape)
      last_dim = tf.compat.dimension_value(input_shape[-1])

      if self.verbose:
         print(last_dim, self.output_channels)

      if last_dim is None:
         Status.CorruptInput()

      self.weight = self.add_weight(name="%s_weight" % self.layer_name,
                                    shape=(last_dim, self.output_channels),
                                    initializer=self.w_initializer,
                                    regularizer=self.b_regularizer,
                                    dtype=self.dtype,
                                    trainable=True)

      if self.add_bias:
         self.bias = self.add_weight(name="%s_bias" % self.layer_name,
                                     shape=(self.output_channels,),
                                     initializer=self.b_initializer,
                                     regularizer=self.b_regularizer,
                                     dtype=self.dtype,
                                     trainable=True)
      else:
         self.bias = None
      self.built = True

   def call(self, input_tensor):

      if self.verbose:
         print("Dense Layer Input: ", input_tensor)

      tensor_rank = input_tensor.shape.rank
      if tensor_rank is None or tensor_rank == 2:
         if isinstance(input_tensor, tf.SparseTensor):
            input_tensor, _ = tf.sparse.fill_empty_rows(inputs, 0)
            ids = tf.SparseTensor(indices=input_tensor.indices, 
                                  values=input_tensor.indices[:, 1], 
                                  dense_shape=input_tensor.dense_shape)
            weights = input_tensor
            outputs = tf.nn.embedding_lookup_sparse(self.weight, ids, weights, combiner='sum')
         else:
            outputs = tf.raw_ops.MatMul(a=input_tensor, b=self.weight)
      else:
         outputs = tf.tensordot(input_tensor, self.weight, [[tensor_rank - 1], [0]])

      if self.add_bias:
         outputs = tf.nn.bias_add(outputs, self.bias)

      if self.activation_fn:
         outputs = self.activation_fn(outputs)

      if self.verbose:
         print("Dense Layer Output: ", outputs)

      return outputs

   def compute_output_shape(self, input_shape):
      input_channels = tf.compat.dimension_value(input_shape[-1])
      output_shape   = (input_channels, self.output_channels)
      return tf.TensorShape(output_shape)

   def get_config(self):
      config = super(DMDense, self).get_config()
      config.update({
         "output_channels" : self.output_channels,
         "w_initializer"   : self.w_initializer,
         "b_initializer"   : self.b_initializer,
         "w_regularizer"   : self.w_regularizer,
         "b_regularizer"   : self.b_regularizer,
         "activity_regularizer" : self.activity_regularizer,
         "activation_fn"   : self.activation_fn,
         "add_bias"        : self.add_bias,
         "verbose"         : self.verbose,
         "name"            : self.layer_name
      })
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class DMGCN(tf.keras.layers.Layer):
   """
   ADMETNet Graph Convolution Layer

   Reference: Kipf & Welling arXiv:1609.02907
   """
   def __init__(self, 
                output_channels:int=32,
                add_bias:bool=True,
                w_initializer:str=Initializer.glorot_normal.name,
                b_initializer:str=Initializer.zeros.name,
                activation_fn:str=None,
                activity_regularizer:str=None,
                w_regularizer:str=None,
                b_regularizer:str=None,
                verbose:bool=False,
                name=None,
                **kwargs):

      self.output_channels = int(output_channels)

      self.w_regularizer        = w_regularizer
      self.b_regularizer        = b_regularizer
      self.activity_regularizer = activity_regularizer

      self.w_initializer = tf.keras.initializers.get(w_initializer)
      self.b_initializer = tf.keras.initializers.get(b_initializer)
      self.activation_fn = tf.keras.activations.get(activation_fn)

      self.add_bias   = add_bias
      self.verbose    = verbose
      self.layer_name = name

      super(DMGCN, self).__init__(name=self.layer_name, 
                                  activity_regularizer=self.activity_regularizer,
                                  **kwargs)


   def build(self, input_shape):
      
      assert len(input_shape) >= 2

      input_dim = input_shape[0][-1]

      if self.verbose:
         print(input_dim)

      self.w = self.add_weight(name="%s_weight" % self.layer_name, 
                               shape=(input_dim, self.output_channels), 
                               initializer=self.w_initializer,
                               regularizer=self.w_regularizer,
                               dtype=self.dtype,
                               trainable=True)

      if self.add_bias:
         self.bias = self.add_weight(name="%s_bias" % self.layer_name,
                                     shape=(self.output_channels,),
                                     initializer=self.b_initializer,
                                     regularizer=self.b_regularizer,
                                     dtype=self.dtype,
                                     trainable=True)
      else:
         self.bias = None
      self.built = True

   def degree_matrix(self, adj_mat):
      D = None
      if kb.is_parse(adj_mat):
         D = tf.sparse.reduce_sum(adj_mat, axis = -1)
      else:
         D = tf.reduce_sum(adj_mat, axis = -1)
      return D

   def call(self, inputs):
      node_features, adj_matrix = inputs 
      
      if self.verbose:
         print("Inputs", inputs)

      """
      Get the inverse root-square of the diagonal
      degree matrix D.
      """
      D = tf.reduce_sum(adj_matrix, axis=-1)
      D = tf.linalg.diag(D)
      D = tf.linalg.inv(D)
      D = tf.linalg.sqrtm(D)

      """
      Updating node_features of the graph.
      """
      H = tf.matmul(node_features, self.w)
      H = tf.matmul(D, H)
      H = tf.matmul(adj_matrix, H)
      H = tf.matmul(D, H)

      output = H

      if self.add_bias:
         output = tf.nn.bias_add(output, self.bias)

      if self.activation_fn:
         output = self.activation_fn(output)

      if self.verbose:
         print("Outputs", output)

      return output, adj_matrix

   def get_config(self):
      config = super(DMGCN, self).get_config()
      config.update({
         "output_channels" : self.output_channels,
         "w_initializer"   : self.w_initializer,
         "b_initializer"   : self.b_initializer,
         "w_regularizer"   : self.w_regularizer,
         "b_regularizer"   : self.b_regularizer,
         "activity_regularizer" : self.activity_regularizer,
         "activation_fn"   : self.activation_fn,
         "add_bias"        : self.add_bias,
         "verbose"         : self.verbose,
         "name"            : self.layer_name
      })
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class DMGAttention(tf.keras.layers.Layer):
   """
   ADMETNet Graph Attention Layer

   Reference: Petar Veličković arXiv:1710.10903
   """
   def __init__(self,
                output_channels:int=16,
                atten_heads:int= 8,
                concat_atten_heads:bool=False,
                add_bias:bool=True,
                dropout_rate:float=0.05,
                w_initializer:str=Initializer.glorot_uniform.name,
                b_initializer:str=Initializer.zeros.name,
                activation_fn:str=None,
                activity_regularizer:str=None,
                w_regularizer:str=None,
                b_regularizer:str=None,
                verbose:bool=True,
                name=None,
                **kwargs):


      self.output_channels = output_channels
      self.atten_heads = atten_heads
      self.concat_atten_heads = concat_atten_heads
      self.dropout_rate = dropout_rate
      self.verbose = verbose
      self.add_bias = add_bias
      self.layer_name = name

      self.w_regularizer        = w_regularizer
      self.b_regularizer        = b_regularizer
      self.activity_regularizer = activity_regularizer

      self.w_initializer = tf.keras.initializers.get(w_initializer)
      self.b_initializer = tf.keras.initializers.get(b_initializer)
      self.activation_fn = tf.keras.activations.get(activation_fn)

      self.w = []
      self.self_attention_w = []
      self.neighbor_attention_w = []
      self.bias = []

      super(DMGAttention, self).__init__(name=self.layer_name, 
                                         activity_regularizer=self.activity_regularizer,
                                         **kwargs)


   def build(self, input_shape):

      assert len(input_shape) >= 2
      input_dim = input_shape[0][-1]

      if self.verbose:
         print(input_dim)

      for head in range(self.atten_heads):
         self.w.append(self.add_weight(name="%s_%d_weight" % (self.layer_name, head), 
                                       shape=(input_dim, self.output_channels), 
                                       initializer=self.w_initializer,
                                       regularizer=self.w_regularizer,
                                       dtype=self.dtype,
                                       trainable=True))

         self.self_attention_w.append(self.add_weight(name="%s_%d_self_attention_weight" % (self.layer_name, head), 
                                                      shape=(self.output_channels, 1), 
                                                      initializer=self.w_initializer,
                                                      regularizer=self.w_regularizer,
                                                      dtype=self.dtype,
                                                      trainable=True))

         self.neighbor_attention_w.append(self.add_weight(name="%s_%d_neighbor_attention_weight" % (self.layer_name, head), 
                                                          shape=(self.output_channels, 1), 
                                                          initializer=self.w_initializer,
                                                          regularizer=self.w_regularizer,
                                                          dtype=self.dtype,
                                                          trainable=True))
         if self.add_bias:
            self.bias.append(self.add_weight(name="%s_%d_bias" % (self.layer_name, head),
                                             shape=(self.output_channels,),
                                             initializer=self.b_initializer,
                                             regularizer=self.b_regularizer,
                                             dtype=self.dtype,
                                             trainable=True))
         else:
            self.bias = None

      self.attention_dropout = DMDropout(rate=self.dropout_rate, name="%s_Attention_Dropout" % self.layer_name)
      self.feature_dropout   = DMDropout(rate=self.dropout_rate, name="%s_Feature_Dropout" % self.layer_name)

      self.built = True
                

   def call(self, inputs):
      X, adj_matrix = inputs

      if self.verbose:
         print(inputs)

      outputs = []
      for head in range(self.atten_heads):

         features = kb.dot(X, self.w[head])

         attention_for_self = kb.dot(features, self.self_attention_w[head])
         attentiion_for_neighbor = kb.dot(features, self.neighbor_attention_w[head])

         atten_coeff = attention_for_self + kb.transpose(attentiion_for_neighbor)

         # Eqn. 3
         atten_coeff = tf.nn.leaky_relu(atten_coeff, alpha=0.2)

         # Injecting the graph strcuture to atten_coeff using a mask matrix
         mask = -10e9 * (1.0 - adj_matrix)
         atten_coeff += mask

         # Eqn. 2, normalizing atten_coeff
         atten_coeff = kb.softmax(atten_coeff)

         dropout_attn = self.attention_dropout(atten_coeff)
         dropout_feat = self.feature_dropout(features)

         node_features = kb.dot(dropout_attn, dropout_feat)

         if self.add_bias:
            node_features = kb.bias_add(node_features, self.bias[head])

         outputs.append(node_features)

      if self.concat_atten_heads:
         # Eqn. 5
         output = tf.concat(outputs)
      else:
         # Eqn. 6
         output = tf.reduce_mean(kb.stack(outputs), axis=0)
         output = output[:,:,0]

      if self.activation_fn:
         # Eqn. 4
         output = self.activation_fn(output)

      if self.verbose:
         print("Outputs", output)

      return output, adj_matrix

   def compute_output_shape(self, input_shape):
      output_shape = input_shape[0][0], self.output_channels
      return output_shape
      
   def get_config(self):
      config = super(DMGAttention, self).get_config()
      config.update({
         "output_channels" : self.output_channels,
         "atten_heads"     : self.atten_heads, 
         "concat_atten_heads" : self.concat_atten_heads,
         "dropout_rate"    : self.dropout_rate,
         "w_initializer"   : self.w_initializer,
         "b_initializer"   : self.b_initializer,
         "activity_regularizer" : self.activity_regularizer,
         "w_regularizer"   : self.w_regularizer,
         "b_regularizer"   : self.b_regularizer,
         "activation_fn"   : self.activation_fn,
         "add_bias"        : self.add_bias,
         "verbose"         : self.verbose,
         "name"            : self.layer_name
      })
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DMActivation(tf.keras.layers.Layer):
   """
   ADMETNet Activation Layer
   """
   def __init__(self, 
                activation_fn:str=ActivationFunction.relu.name, 
                name=None,
                **kwargs):

      super(DMActivation, self).__init__(name=name, **kwargs)

      self.activation_fn = tf.keras.layers.Activation(activation_fn) 
      self.supports_masking = True

   def call(self, inputs):
      outputs = self.activation_fn(inputs)
      return outputs

   def compute_output_shape(self, input_shape):
      return input_shape

   def get_config(self):
      config = super(DMActivation, self).get_config()
      config.update({
         "activation_fn" : self.activation_fn,
      })
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class DMDropout(tf.keras.layers.Layer):
   """
   ADMETNet Dropout Layer
   """
   def __init__(self, 
                rate:float=0.05, 
                name=None, 
                **kwargs):

      super(DMDropout, self).__init__(name=name, **kwargs)

      self.rate:float = rate

   def call(self, inputs, training=True):
      if training:
         return tf.nn.dropout(inputs, rate=self.rate)
      return inputs

   def get_config(self):
      config = super(DMDropout, self).get_config()
      config.update({
         "rate":self.rate,
      })
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class DMGReduce(tf.keras.layers.Layer):
   """
   ADMETNet Graph Reduction Layer
   """
   def __init__(self, 
                aggregation_fn=None, 
                name=None, 
                **kwargs):

      super(DMGReduce, self).__init__(name=name, **kwargs)

      self.aggregation_fn = aggregation_fn

   def call(self, inputs):
      nodes, adj = inputs
      reduction = tf.reduce_mean(nodes, axis=1)
      return reduction

   def get_config(self):
      config = super(DMGReduce, self).get_config()
      config.update({
         "aggregation_fn": self.aggregation_fn,
      })
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)

