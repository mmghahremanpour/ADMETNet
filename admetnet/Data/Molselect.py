
from admetnet.Utils.Imports import *


class Molselect:
   
   def __init__(self):
      
      self.molset    = None
      self.shuffle   = False
      self.bootstrap = True

      self.random_state = None
      self.ratio = 0.8

   def set_molset(self, molecules):
      self.molset = molecules

   def get_molset(self):
      return self.molset

   def set_shuffle(self, shuffle):
      self.shuffle = shuffle

   def get_shuffle(self):
      return self.shuffle

   def set_bootstrap(self, bootstrap):
      self.bootstrap = bootstrap

   def get_bootstrap(self):
      return self.bootstrap

   def set_random_state(self, seed):
      self.random_state = seed

   def set_ratio(self, ratio):
      self.ratio = ratio

   def ratio(self):
      return self.ratio

   def get_random_state(self):
      return self.random_state

   def bootsratpped_sample(self):

      random_state = np.random.RandomState()

      if self.random_state:
         random_state = np.random.RandomState(seed=self.random_state)

      if self.shuffle:
         random_state.shuffle(self.molset)

      # bootstrapped observations
      training_set = random_state.choice(self.molset, 
                                         size=len(self.molset), 
                                         replace=True)

      # out of bag observations
      test_set = [mol for mol in self.molset if mol not in training_set]

      for tr in training_set:
         tr.set_train_set(True)

      for ts in test_set:
         ts.set_train_set(False)

      return training_set, test_set

   def random_sample(self):

      random_state = np.random.RandomState()

      if self.random_state:
         random_state = np.random.RandomState(seed=self.random_state)

      if self.shuffle:
         random_state.shuffle(self.molset)

      training_set = []
      test_set = []
      for molecule in self.molset:
         if random_state.uniform(0, 1) <= self.ratio:
            molecule.set_train_set(True)
            training_set.append(molecule)
         else:
            molecule.set_train_set(False) 
            test_set.append(molecule)

      return training_set, test_set

   def make_selection(self):

      if self.bootstrap:
         return self.bootsratpped_sample()
      else:
         return self.random_sample()
