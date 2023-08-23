
from admetnet.Utils.Imports     import *
from admetnet.Utils.Commandline import *
from main.estimator             import *

ADMETNET = os.environ["ADMETNET"]

GCN = "%s/pretrained/logS/GCN/" % ADMETNET
GAT = "%s/pretrained/logS/GAT/" % ADMETNET

def solnet():
   args      = commandline_options()
   config    = config_args(args)

   if args.network_type == NetworkType.GCN.name:
      config["Estimators_PATH"] = GCN
   elif args.network_type == NetworkType.GAT.name:
      config["Estimators_PATH"] = GAT
   else:
      sys.exit("Not supported network")

   estimates = estimator(config)


   if estimates:
      (molecules, mean, std) = estimates
      out = open(args.output_file, "w")
      for i, molecule in enumerate(molecules):
         out.write("%s, %s, %0.3f, %0.3f\n" % (molecule.name, molecule.smiles, mean[i], std[i]))
      out.close()
   else:
      print("Cound not predict logS.")

if __name__ == "__main__":
   solnet()
