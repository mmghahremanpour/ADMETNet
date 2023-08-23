

import sys
import math

if "copy" not in sys.modules:
   import copy
if "time" not in sys.modules:
   import time
if "tf" not in sys.modules:
   import tensorflow as tf
if "kb" not in sys.modules:
   import tensorflow.keras.backend as kb
if "os" not in sys.modules: 
   import os
if "json" not in sys.modules: 
   import json
if "openbabel" not in sys.modules:
   import openbabel
if "AllChem" not in sys.modules:
   import rdkit.Chem.AllChem as AllChem
if "Chem" not in sys.modules:
   import rdkit.Chem as Chem
if "EState" not in sys.modules:
   import rdkit.Chem.EState as EState
if "DataStructs" not in sys.modules:
   from rdkit import DataStructs
if "Butina" not in sys.modules:
   from rdkit.ML.Cluster import Butina
if "np" not in sys.modules:
   import numpy     as np
if "deepcopy" not in sys.modules:
   from copy        import deepcopy
if "Path" not in sys.modules:
   from pathlib     import Path
if "Popen" not in sys.modules: 
   from subprocess  import Popen, PIPE
if "OrderedDict" not in sys.modules: 
   from collections import OrderedDict, defaultdict, abc
if "combinations_with_replacement" not in sys.modules:
   from itertools   import combinations_with_replacement, permutations
if "ArgumentParser" not in sys.modules:
   from argparse    import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
if "rmtree" not in sys.modules:
   from shutil      import copyfile, rmtree, which
if "dataclass" not in sys.modules:
   from dataclasses import dataclass, replace, is_dataclass
if "Enum" not in sys.modules:
   from enum        import Enum, unique, EnumMeta
if "pickle" not in sys.modules:
   import pickle
if "TypeVar" not in sys.modules:
   from typing import TypeVar, Union, List, Tuple, Iterable, Dict, NoReturn, Callable
if "cross_val_score" not in sys.modules:
   from sklearn.model_selection import cross_val_score
if "RepeatedKFold" not in sys.modules:
   from sklearn.model_selection import RepeatedKFold
if "make_regression" not in sys.modules:
   from sklearn.datasets import make_regression
if "sk_en" not in sys.modules:
   import sklearn.ensemble as sk_en
if "sk_nn" not in sys.modules:
   import sklearn.neural_network as sk_nn
if "seaborn" not in sys.modules:
   import seaborn
if "random" not in sys.modules:
   import random
if "plt" not in sys.modules:
   import matplotlib.pyplot as plt

os = sys.modules["os"]
json = sys.modules["json"]
pickle = sys.modules["pickle"] 
random = sys.modules["random"]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

numThreads = int(os.getenv('SLURM_CPUS_PER_TASK',1))
numInterOpThreads = 1

assert numThreads % numInterOpThreads == 0

numIntraOpThreads = numThreads // numInterOpThreads

os.environ['OMP_NUM_THREADS'] = str(numIntraOpThreads)

tf.config.threading.set_intra_op_parallelism_threads(numIntraOpThreads)
tf.config.threading.set_inter_op_parallelism_threads(numIntraOpThreads)

