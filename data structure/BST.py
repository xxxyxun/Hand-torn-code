class TreeNode():
    def __init__(self, key, val, left= None, right= None, parent= None):
        self.val = val
        self.key = key
        self.left = left
        self.right = right 
        self.parent = parent

    def __str__(self):
        return str(self.val)
    
    def hasLeft(self):
        return self.left
    
    def hasRight(self):
        return self.right
    
    def isLeft(self):
        return self.parent and self.parent.left == self
    
    def isRight(self):
        return self.parent and self.parent.right == self
    
    def isroot(self):
        return not self.parent
    
    def isleafroot(self):
        return not (self.left or self.right)
    
    def hasanychild(self):
        return self.left or self.right
    
    def hasbothchild(self):
        return self.left and self.right
    
    def replaceNodeDota(self, key, value, lc, rc):
        self.key = key
        self.val = value
        self.left = lc
        self.right = rc
        if not self.right:
            self.right.parent = self
        if not self.left:
            self.left.parent = self
    
    

class BinarySearchTree():
    def __init__(self, node=None):
        self.root = node
        self.size = 0

    def length(self):
        return self.size
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        return self.root.__iter__()
    

    def insert(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val)
        self.size += 1
    
    def _put(self, key, val, cur: TreeNode):
        if key < cur.key:
            if cur.left:
                self._put(key, val, cur.left)
            else:
                cur.left = TreeNode(key, val, parent=cur)
        else:
            if cur.right:
                self._put(key, val, cur.right)
            else:
                cur.right = TreeNode(key, val, parent=cur)

    def __setitem__(self, key, val):
        return self.insert(key, val)

#返回对应键的值
    def get(self, key):
        if self.root:
            res = self._get(key,self.root)
            if res:
                return res.val
            else:
                return None
        else:
            return None
    
    def _get(self, key, cur: TreeNode):
        if not cur:
            return None
        elif cur.key == key:
            return cur
        elif cur.key > key:
            return self._get(key, cur.left)
        else:
            return self._get(key, cur.right)

    def __getitem__(self, key):
        return self.get(key)
    
    def __contains__(self, key):
        if self._get(key,self.root):
            return True
        else:
            return False
if __name__ == '__main__':
    bst = BinarySearchTree()
    bst[17] = "tiger"
    bst[26] = "dog"
    bst[31] = "cow"
    bst[54] = "cat"
    bst[93] = "lion"
    bst[77] = "bird"
    print(bst[77])



        