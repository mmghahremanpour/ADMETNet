
class Error(Exception):
   pass

class NoMoleculeInTrainingData(Error):
   pass

class MoleculeWithNoSMILE(Error):
   pass

class NotSupportedProperty(Error):
   pass

class NotSupportedTrainer(Error):
   pass

class RdKitMolNotGenerated(Error):
   pass

class OBMolNotGenerated(Error):
   pass

class PropExists(Error):
   pass

class PropDoseNotExist(Error):
   pass

class AtomExists(Error):
   pass

class OutOfAtomIndex(Error):
   pass

class ForceFieldNotFound(Error):
   pass

class AtomExists(Error):
   pass

class MolGraphError(Error):
   pass

class NoMolculeInData(Error):
   pass

