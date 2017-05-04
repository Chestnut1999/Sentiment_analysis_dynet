class TreeNode:
	# rootval : string ( a word )
	# rootid: int
	# rootid starts from 1, root node has id = 0
	# rootvector: a cnn expression vector
	def __init__(self, rootvector = None, rootval = None, rootid = None, isleaf = False):
		self.id = rootid 
		self.val = [rootval]
		self.parent = None
		#self.level = 0
		self.vector = rootvector
		self.children = []
		self.isleaf = isleaf
	def insert_parent(self, parentnode):
		self.parent = parentnode
		parentnode.children.append(self)
		#elf.level += 1
	def update_vector(self, vector):
		self.vector = vector

