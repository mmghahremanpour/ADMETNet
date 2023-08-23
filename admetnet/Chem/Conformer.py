
from deepmodeller.Utils.Status  import * 
from deepmodeller.Utils.Imports import *

class ConfGen:

   def __init__(self, rdMol=None, SDFfile=None):

      self._rdMol   = rdMol
      self._SDFfile = SDFfile

   def set_rdMol(self, rdMol):
      self._rdMol = rdMol

   def set_filename(self, filename):
      self._SDFfile = filename

   def _confgen_SDFwriter(self, 
                          clusters, 
                          conformers_prop, 
                          minimum_energy:float=0.0):

      if self._SDFfile is None:
         Status.SDFfileNeeded()

      sdf = Chem.SDWriter(self._SDFfile)
      for propname in self._rdMol.GetPropNames():
         self._rdMol.ClearProp(propname)
      for cluster in clusters:
         for idx in cluster:
            self._rdMol.SetIntProp("Conformer_id", idx+1)
            conformer_prop = conformers_prop[idx]
            for k, v in conformer_prop.items():
               self._rdMol.SetProp(k, str(v))
            conformer_energy = conformer_prop["energy"]
            if conformer_energy:
               self._rdMol.SetDoubleProp("Relative Energy", conformer_energy - minimum_energy)
            sdf.write(self._rdMol, confId=idx)
      sdf.flush()
      sdf.close()
               
   def _confgen_energy(self, 
                       conformer_id:int=1, 
                       maxiter:int=100):

      ff = AllChem.MMFFGetMoleculeForceField(self._rdMol, 
                                             AllChem.MMFFGetMoleculeProperties(self._rdMol), 
                                             confId=conformer_id)
      ff.Initialize()
      ff.CalcEnergy()
      energy = dict()
      if maxiter:
         energy["converged"] = ff.Minimize(maxIts=maxiter)
      energy["energy"] = ff.CalcEnergy()
      return energy

   def _confgen_cluster(self, mode="RMSD", epsilon=2.0):

      distane_matrix = None
      if mode == "TFD":
         distance_matrix = Chem.TorsionFingerprints.GetTFDMatrix(self.rdMol)
      else:
         distance_matrix = AllChem.GetConformerRMSMatrix(self._rdMol, prealigned=False)
      clusters = Butina.ClusterData(distance_matrix, 
                                    self._rdMol.GetNumConformers(), 
                                    epsilon, 
                                    isDistData=True, 
                                    reordering=True)
      return clusters

   def _confgen_aligner(self, cluster):
      RMSlist = list()
      AllChem.AlignMolConformers(self._rdMol, confIds=cluster, RMSlist=RMSlist)
      return RMSlist

   def _conformer_generator(self, 
                            nconf:int=10, 
                            random:bool=True):
      """
      This function generates conformations for a molecule. 
      It is yet to be implemented. 
      """
      try:
         if self._rdMol:
            params = AllChem.ETKDGv3() 
            params.useSmallRingTorsions=True
            if random:
               params.useRandomCoords=True
            ids = AllChem.EmbedMultipleConfs(self._rdMol, numConfs=nconf, params=params)
            return list(ids)
         else:
            raise RdKitMolNotGenerated
      except RdKitMolNotGenerated:
         Status.RdKitMolNotGenerate()

   def confgen(self,
               nconf:int=10,
               random:bool=True,
               minimum_energy:float=1e+34):

      n = 1
      conformers_prop = dict()
      for conformer_id in self._conformer_generator(nconf=nconf, random=random):
         conformers_prop[conformer_id] = self._confgen_energy(conformer_id=conformer_id)
      clusters = self._confgen_cluster()
      for cluster in clusters:
         rms_within_cluster = self._confgen_aligner(cluster=cluster) 
         for conformer_id in cluster:
            props  = conformers_prop[conformer_id]
            energy = props["energy"]
            if energy < minimum_energy:
               minimum_energy = energy
            props["cluster_no"] = n
            props["cluster_centroid"] = cluster[0] + 1
            idx = cluster.index(conformer_id)
            if idx:
               props["rms_to_centroid"] = rms_within_cluster[idx-1]
            else:
               props["rms_to_centroid"] = 0.0
         n += 1
      self._confgen_SDFwriter(clusters=clusters, 
                              conformers_prop=conformers_prop, 
                              minimum_energy=minimum_energy)

