"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Utils.Imports   import *
from admetnet.Utils.GracePlot import *
from IPython.display import clear_output

class LearningRateScheduler(tf.keras.callbacks.Callback):
   def __init__(self, schedule):
      super(LearningRateScheduler, self).__init__()
      self.schedule = schedule

   def on_epoch_begin(self, epoch, logs=None):
      if not hasattr(self.model.optimizer, "lr"):
         raise ValueError("Optimizer dose not have learning rate lr attribute")
      lr = float(kb.get_value(self.model.optimizer.learning_rate))
      scheduler_lr = self.schedule(epoch, lr)
      kb.set_value(self.model.optimizer.lr, scheduler_lr)
      print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

class DMLogger(tf.keras.callbacks.Callback):

  def __init__(self, filename, separator=',', append=False):
    self.sep = separator
    self.filename = filename #path_to_string(filename)
    self.append = append
    self.writer = None
    self.keys = None
    self.append_header = True
    super(DMLogger, self).__init__()

  def on_train_begin(self, logs=None):
    if self.append:
      if tf.io.gfile.exists(self.filename):
        with tf.io.gfile.GFile(self.filename, 'r') as f:
          self.append_header = not bool(len(f.readline()))
      mode = 'a'
    else:
      mode = 'w'
    self.csv_file = tf.io.gfile.GFile(self.filename, mode)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}

    def handle_value(k):
      is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
      if isinstance(k, str):
        return k
      elif isinstance(k, abc.Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
      else:
        return k

    if self.keys is None:
      self.keys = sorted(logs.keys())

    if self.model.stop_training:
      # We set NA so that csv parsers do not fail for this last epoch.
      logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

    if not self.writer:

      import csv
      class CustomDialect(csv.excel):
        delimiter = self.sep

      fieldnames = ['epoch'] + self.keys

      self.writer = csv.DictWriter(
          self.csv_file,
          fieldnames=fieldnames,
          dialect=CustomDialect)
      if self.append_header:
        self.writer.writeheader()

    row_dict = OrderedDict({'epoch': epoch})
    row_dict.update((key, handle_value(logs[key])) for key in self.keys)
    self.writer.writerow(row_dict)
    self.csv_file.flush()

  def on_train_end(self, logs=None):
    self.csv_file.close()
    self.writer = None


class GracePlotter(tf.keras.callbacks.Callback):

   def __init__(self, filename):
      self.filename = filename #path_to_string(filename)
      self.keys = None
      super(GracePlotter, self).__init__()

   def on_train_begin(self, logs=None):
      self.loss_on_train = list()
      self.loss_on_test  = list()

   def on_epoch_end(self, epoch, logs=None):
      logs = logs or {}

      def handle_value(k):
         is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
         if isinstance(k, str):
            return k
         elif isinstance(k, abc.Iterable) and not is_zero_dim_ndarray:
            return '"[%s]"' % (', '.join(map(str, k)))
         else:
            return k

      if self.keys is None:
         self.keys = sorted(logs.keys())

      if self.model.stop_training:
         logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

      row_dict = OrderedDict({'epoch': epoch})
      row_dict.update((key, handle_value(logs[key])) for key in self.keys)
      self.loss_on_train.append(row_dict["loss"])
      self.loss_on_test.append(row_dict["val_loss"])

   def on_train_end(self, logs=None):
      self.xvg_file = open(self.filename, "w")
      self.xvg_file.write("%s\n" % xmgrace_params.learning_curve)
      self.xvg_file.write("@target G0.S0\n")
      self.xvg_file.write("@type xy\n")
      for i, value in enumerate(self.loss_on_train):
         self.xvg_file.write("%d %0.4f\n" % (i, value))
      self.xvg_file.write("&\n")
      self.xvg_file.write("@target G0.S1\n")
      self.xvg_file.write("@type xy\n")
      for i, value in enumerate(self.loss_on_test):
         self.xvg_file.write("%d %0.4f\n" % (i, value))
      self.xvg_file.write("&\n")
      self.xvg_file.write("@autoscale\n")
      self.xvg_file.close()

   def regression(self):
      pass

   def classification(self):
      pass


class RealTimePlot(tf.keras.callbacks.Callback):

   def __init__(self):
      self.keys = None
      super(RealTimePlot, self).__init__()

   def on_train_begin(self, logs=None):
      self.loss_on_train = list()
      self.loss_on_test  = list()

   def on_epoch_end(self, epoch, logs=None):
      logs = logs or {}

      def handle_value(k):
         is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
         if isinstance(k, str):
            return k
         elif isinstance(k, abc.Iterable) and not is_zero_dim_ndarray:
            return '"[%s]"' % (', '.join(map(str, k)))
         else:
            return k

      if self.keys is None:
         self.keys = sorted(logs.keys())

      if self.model.stop_training:
         logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

      row_dict = OrderedDict({'epoch': epoch})
      row_dict.update((key, handle_value(logs[key])) for key in self.keys)
      self.loss_on_train.append(row_dict["loss"])
      self.loss_on_test.append(row_dict["val_loss"])

      clear_output(wait=True)
      N = np.arange(0, len(self.loss_on_train))
      
      plt.style.use("ggplot")

      plt.figure(figsize=(10,3))
      plt.plot(N, self.loss_on_train, label="Training Set")
      plt.plot(N, self.loss_on_test, label="Test Set")
      plt.legend()
      plt.title("Training Curve")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.show()

   def on_train_end(self, logs=None):
      plt.close()
