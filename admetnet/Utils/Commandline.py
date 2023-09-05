"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Utils.Imports import *
 

def int_range(mini,maxi):
   def int_range_checker(arg):
      try:
         i = int(arg)
      except ValueError:
         raise ArgumentTypeError("must be an integer")
      if i < mini or i > maxi:
         raise ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
      return i
   return int_range_checker
         

def commandline_options():
   parser = ArgumentParser(
           prog='ADMETNet',
           formatter_class=RawDescriptionHelpFormatter,
           description="""
            ADMETNet Version 1.0
          Mohammad M. Ghahremanpour
      William L. Jorgensen Research Group
             Yale University
                 2020-2023

  """
  )
   parser.add_argument("-if",  "--input_file",     help="Input file in .csv format",                   type=str,   default=None)
   parser.add_argument("-of",  "--output_file",    help="Output file in .csv format",                  type=str,   default="solnet_output.csv")
   parser.add_argument("-is",  "--input_smiles",   help="Input SMILES",                                type=str,   default=None)
   parser.add_argument("-nt",  "--network_type",   help="Type of Neural Network: GCN, GAT",            type=str,   default="GCN")
   parser.add_argument("-ne",  "--n_estimator",    help="Number of estimators",                        type=int_range(1,30),   default=30)
   parser.add_argument("-dbp", "--database_paramfile", help="Database file in .json format",           type=str,   default=None)
   args = parser.parse_args()
   return args
