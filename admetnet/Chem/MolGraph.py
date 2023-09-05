"""
This source file is part of the ADMETNet package.

Developer:
    Mohammad M. Ghahremanpour
    William L. Jorgensen Research Group
    Chemistry Department
    Yale University
"""

from admetnet.Utils.Imports  import *
from admetnet.Utils.Lambdas  import *
from admetnet.Utils.Status   import *
from admetnet.Utils.Exceptions import *

class GBase:
   """
   Base class inherited to all classes related 
   to graph representation of a molecule.
   """
   def __init__(self):
      self._idx:int=None
      self._attributes=None
      self._node_attribute_matrix=None
      self._edge_attribute_matrix=None
      self._attribute_tensor=None
      self._adjacency_matrix=None
      self._adjacency_vector=None
      self._laplacian_matrix=None
      self._normalized_laplacian_matrix=None
      self._node_degree_vector=None

      self._NODE_ATTRIBUTES = {}
      self._NODE_ATTRIBUTES["AtomicNumber"]     = [1,5,6,7,8,9,14,15,16,17,35,53,"Other"]
      self._NODE_ATTRIBUTES["Hybridization"]    = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
      self._NODE_ATTRIBUTES["ImplicitValence"]  = [0,1,2,3,4,5,6]
      self._NODE_ATTRIBUTES["Degree"]           = [0,1,2,3,4,5,6]
      self._NODE_ATTRIBUTES["TotalNumHydrogen"] = [0,1,2,3,4]
      self._NODE_ATTRIBUTES["FormalCharge"]     = [-3,-2,-1,0,1,2,3]

      self._EDGE_ATTRIBUTES = {}
      self._EDGE_ATTRIBUTES["BondOrder"]  = [1.0, 1.5, 2.0, 3.0]
      self._EDGE_ATTRIBUTES["BondStereo"] = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]

   def get_attributes(self)->List:
      return self._attributes

   def set_attributes(self,  attributes=list())->NoReturn:
      self._attributes = nparf32(attributes)

   def n_attributes(self):
      return self._attributes.size

   def graph_attr_tensor(self):
      return self._attribute_tensor

   def node_deg_vector(self):
      return self._node_degree_vector

   def adj_matrix(self):
      return self._adjacency_matrix 

   def laplacian_matrix(self):
      return self._laplacian_matrix

   def normalized_laplacian_matrix(self):
      return self._normalized_laplacian_matrix

   def node_edge_attr_matrix(self):
      return (self._node_attribute_matrix, self._edge_attribute_matrix)

   def node_attr_adj_matrix(self):
      return (self._node_attribute_matrix, self._adjacency_matrix)

   @property
   def index(self)->int:
      return self._idx

   @index.setter
   def index(self, idx:int=0):
      self._idx = idx

class Node(GBase):
   def __init__(self):
      GBase.__init__(self)
      self._neighbors:List[int] = list()
      self._degree:int=None
      self._atom_index:int=None
      self._node_index:int=None

      self._dmAtom = None

   def __eq__(self, other):
      return self.degree() == other.degree()

   def __lt__(self, other):
      return self.degree() < other.degree()

   def neighbors(self)->List[int]:
      return self._neighbors

   def append_neighbors(self, atom_index, bond_index)->NoReturn:
      self._neighbors.append((atom_index, bond_index))

   def degree(self):
      return self._degree

   @property
   def node_index(self):
      return self._node_index

   @node_index.setter
   def node_index(self, index):
      self._node_index = index

   @property
   def atom_index(self):
      return self._atom_index

   @atom_index.setter
   def atom_index(self, index):
      self._atom_index = index

   def atom_attributes(self, 
                       rdAtom=None, 
                       features=None, 
                       atom_descriptors=None)->NoReturn:

      attributes:List[float]  = list()

      if features["atomic_number"]:
         attributes += hotvec_unk(rdAtom.GetAtomicNum(),  
         self._NODE_ATTRIBUTES["AtomicNumber"]) 

      if features["atomic_hybridization"]:
         attributes += hotvec_unk(rdAtom.GetHybridization(), 
         self._NODE_ATTRIBUTES["Hybridization"])
   
      if features["atom_degree"]:
         attributes += hotvec_unk(rdAtom.GetDegree(), 
         self._NODE_ATTRIBUTES["Degree"]) 

      if features["implicit_valence"]:
         attributes += hotvec_unk(rdAtom.GetImplicitValence(), 
         self._NODE_ATTRIBUTES["ImplicitValence"]) 

      if features["number_of_hydrogens"]:
         attributes += hotvec_unk(rdAtom.GetTotalNumHs(), 
         self._NODE_ATTRIBUTES["TotalNumHydrogen"]) 

      if features["formal_charge"]:
         attributes += hotvec_unk(rdAtom.GetFormalCharge(), 
         self._NODE_ATTRIBUTES["FormalCharge"])

      if features["ring_size"]:
         attributes.append(rdAtom.IsInRing())
         attributes.append(rdAtom.IsInRingSize(3))
         attributes.append(rdAtom.IsInRingSize(4))
         attributes.append(rdAtom.IsInRingSize(5))
         attributes.append(rdAtom.IsInRingSize(6))
         attributes.append(rdAtom.IsInRingSize(7))
         attributes.append(rdAtom.IsInRingSize(8))

      if features["aromaticity"]:
         attributes.append(rdAtom.GetIsAromatic())

      if atom_descriptors:
         attributes += atom_descriptors

      self.set_attributes(attributes)

      self._degree     = rdAtom.GetDegree()
      self._atom_index = rdAtom.GetIdx()
      
class Edge(GBase):
   def __init__(self):
      GBase.__init__(self)
      self._connects = tuple() 

   def connects(self)->Tuple:
      return self._connects

   def append_connects(self, atom1_index, atom2_index)->NoReturn:
      self._connects = (atom1_index, atom2_index)

   def bond_attributes(self, rdBond=None, features=None)->NoReturn:

      attributes:List[float]  = list()

      if features["bond_order"]:
         attributes += hotvec_unk(rdBond.GetBondTypeAsDouble(), 
         self._EDGE_ATTRIBUTES["BondOrder"])

      if features["chirality"]:   
         attributes += hotvec_unk(rdBond.GetStereo(), 
         self._EDGE_ATTRIBUTES["BondStereo"])

      if features["aromaticity"]:
         attributes.append(rdBond.GetIsAromatic())

      if features["bond_conjugation"]:
         attributes.append(rdBond.GetIsConjugated())

      if features["ring_size"]:
         attributes.append(rdBond.IsInRing())

      attributes.append(1) # For Tensor Representation
      self.set_attributes(attributes) 

class MolGraph(GBase):
   """
   Undirected Graph for Describing a Molecule.
   Reference: J. Chem. Inf. Model., 2017, 57, 1757−1772
   """

   def __init__(self):
      GBase.__init__(self)

      self.nodes:List[Node] = list()
      self.edges:List[Edge] = list()
      self.num_nodes:int = 0
      self.num_edges:int = 0
      self.num_attributes:int = 0
      self.nodes_are_sorted:bool = False

      self.adj_matrix_is_built = False
      self.attr_tensor_is_built = False
      self.node_attr_matrix_is_built = False
      self.laplacian_matrix_is_built = False

   def n_nodes(self)->int:
      return self.num_nodes

   def n_edges(self)->int:
      return self.num_edges

   def _calc_total_attributes(self)->NoReturn:
      if self.nodes:
         self.num_attributes += self.nodes[0].n_attributes()
      if self.edges:
         self.num_attributes += self.edges[0].n_attributes()

   def num_total_attributes(self)->int:
      return self.num_attributes

   def append_node(self, node)->NoReturn:
      self.nodes.append(node)
      self.num_nodes += 1

   def sort_nodes_by_degree(self):
      self.nodes = sorted(self.nodes)
      for i, node in enumerate(self.nodes):
         node.node_index = i
      self.nodes_are_sorted = True

   def map_atom_to_node_index(self, index):
      for node in self.nodes:
         if node.atom_index == index:
            return node.node_index
      return None

   def append_edge(self, edge)->NoReturn:
      self.edges.append(edge)
      self.num_edges += 1

   def node_neighbors(self)->List[int]:
      """
      Returns 1D array
      """
      return [n.neighbors() for n in self.nodes]

   def node_attributes(self)->List[float]:
      """
      Returns 2D array as numpy vertical stack 
      """ 
      return np.vstack([n.get_attributes() for n in self.nodes])

   def edge_attributes(self)->List[float]:
      """
      Returns 2D array as numpy vertical stack
      """ 
      return np.vstack([e.get_attributes() for e in self.edges])

   def node_attributes_kb(self)->List[float]:
      """
      Returns 2D array as Keras Backend Variable
      """
      return kb.variable(np.vstack([n.get_attributes() for n in self.nodes]))

   def edge_attributes_kb(self)->List[float]:
      """
      Returns 2D array as Keras Backend Variable
      """
      return kb.variable(np.vstack([e.get_attributes() for e in self.edges]))

   def _check_graph(self)->bool:
      if self.num_nodes == 0:
         raise(MolGraphError, "MolGraph should have at least one node!")
      if self.num_nodes > 1 and self.num_edges == 0:
         #raise(MolGraphError, "MolGraph has %d nodes but no edge!!" % self.num_nodes)
         sys.exit("MolGraph has %d nodes but no edge!!" % self.num_nodes)
      return True

   def make_graph_attr_tensor(self, max_num_node=None):

      if self.attr_tensor_is_built:
         return 

      if self._check_graph():

         N = self.num_nodes
         if max_num_node:
            if max_num_node < N:
               Status.WrongSizeForAttrMatrix()
            N = max_num_node

         self._calc_total_attributes()
         F = self.num_total_attributes()

         self._attribute_tensor = np.zeros((N, N, F))

         if self.num_nodes == 1 and self.num_edges == 0:
            self._attribute_tensor[0, 0, :] = self.nodes[0].get_attributes()
         else:
            edge_attributes = self.edge_attributes()
            node_attributes = self.node_attributes()

            for i, node in enumerate(self.nodes):
               self._attribute_tensor[i, i, :] = np.concatenate((node_attributes[i], np.zeros_like(edge_attributes[0])))

            for k, edge in enumerate(self.edges):
               (i,j) = edge.connects()

               if self.nodes_are_sorted: 
                  i = self.map_atom_to_node_index(i)
                  j = self.map_atom_to_node_index(j)

               self._attribute_tensor[i, j, :] = np.concatenate((node_attributes[j], edge_attributes[k]))
               self._attribute_tensor[j, i, :] = np.concatenate((node_attributes[i], edge_attributes[k]))

         self.attr_tensor_is_built = True

   def make_node_edge_attr_matrix(self, max_num_node=None):
  
      if self.node_attr_matrix_is_built:
         return 

      if self._check_graph():

         N = self.num_nodes
         if max_num_node:
            if max_num_node < N:
               Status.WrongSizeForAttrMatrix()
            N = max_num_node

         F = self.nodes[0].n_attributes()
         self._node_attribute_matrix = np.zeros((N, F))
         node_attributes = self.node_attributes()
         for i, node in enumerate(self.nodes):
            self._node_attribute_matrix[i, :] = node_attributes[i]

         if self.edges:
            F = self.edges[0].n_attributes()
            self._edge_attribute_matrix = np.zeros((N, F))
            edge_attributes = self.edge_attributes()
            for e, edge in enumerate(self.edges):
               (i,j) = edge.connects()

               if self.nodes_are_sorted: 
                  i = self.map_atom_to_node_index(i)
                  j = self.map_atom_to_node_index(j)

               self._edge_attribute_matrix[i, :] += edge_attributes[e]
               self._edge_attribute_matrix[j, :] += edge_attributes[e]

         self.node_attr_matrix_is_built = True
            
   def make_adjacency_matrix(self, 
                             max_num_node:int=None, 
                             add_edge_attr=False):

      if self.adj_matrix_is_built:
         return
   
      if self._check_graph():

         if not self.edges:
            Status.NoEdgeforAdjMatrix()

         N = self.num_nodes
         if max_num_node:
            if max_num_node < N:
               Status.WrongSizeForAttrMatrix()
            N = max_num_node

         self._adjacency_matrix = np.zeros((N,N))
         if add_edge_attr:
            F = self.edges[0].n_attributes()
            self._adjacency_matrix = np.zeros((N,N,F))

         self._adjacency_vector = [[] for _ in range(N)]
         self._node_degree_vector = [0]*N
         
         for i, node in enumerate(self.nodes):
            self._adjacency_matrix[i, i] = 1.0
            self._node_degree_vector[i] = node.degree()

         edge_attributes = self.edge_attributes()
         for e, edge in enumerate(self.edges):
            (i,j) = edge.connects()
        
            if self.nodes_are_sorted: 
               i = self.map_atom_to_node_index(i)
               j = self.map_atom_to_node_index(j)

            self._adjacency_vector[i].append(j)
            self._adjacency_vector[j].append(i)

            if add_edge_attr:
               self._adjacency_matrix[i, j, :] = edge_attributes[e]
               self._adjacency_matrix[j, i, :] = edge_attributes[e]
            else:
               self._adjacency_matrix[i, j] = 1.0
               self._adjacency_matrix[j, i] = 1.0

         self.adj_matrix_is_built = True

   def make_laplacian_matrix(self):
      if not self.adj_matrix_is_built:
         self.make_adjacency_matrix()
      
      A = self._adjacency_matrix
      D = np.sum(A, axis=-1)
      D = np.diag(D)
      L = D - A

      self._laplacian_matrix = L

   def make_normalized_laplacian_matrix(self):
      if not self.adj_matrix_is_built:
         self.make_adjacency_matrix()
      
      A = self._adjacency_matrix
      D = np.sum(A, axis=-1)
      D = np.diag((D + 1e-5)**(-0.5))
      L = np.identity(self.num_nodes) - np.dot(D,A).dot(D) 

      self._normalized_laplacian_matrix = L

   def convolute(self, 
                 max_num_node:int=None, 
                 add_edge_attr:bool=False, 
                 master_atom:bool=False)->NoReturn:
     
      """Sort nodes from low to high degree """ 
      if not self.nodes_are_sorted:
         self.sort_nodes_by_degree()

      self.make_node_edge_attr_matrix(max_num_node=max_num_node)

      self.make_adjacency_matrix(max_num_node=max_num_node, 
                                 add_edge_attr=add_edge_attr)  

      """Sort adj vector from low to high degree """
      if self.nodes_are_sorted:
         self._adjacency_vector.sort(key=len)

      if master_atom:
         master_atom_index = self.num_nodes
         master_atom_features = np.expand_dims(np.mean(self._node_attribute_matrix, axis=0), axis=0)
         self._node_attribute_matrix = np.concatenate([self._node_attribute_matrix, master_atom_features], axis=0)
         for index in range(self.num_nodes):
            self._adjacency_vector[index].append(master_atom_index)

      return ConvMolGraph(self._node_attribute_matrix, 
                          self._node_degree_vector,
                          self._adjacency_matrix, 
                          self._adjacency_vector)
         
