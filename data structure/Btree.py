class TreeNode():
    def __init__(self, data, left=None, right=None) -> None:
        self.data = data
        self.leftroot = left
        self.rightroot = right

    def insert(self, data):
        if self.data:
            if data < self.data:
                if self.leftroot:
                    self.leftroot.insert(data)
                else:
                    self.leftroot = TreeNode(data)
            else:
                if self.rightroot:
                    self.rightroot.insert(data)
                else:
                    self.rightroot = TreeNode(data)
        else:
            self.data = data
    def PrintTree(self):
         if self.leftroot:
             self.leftroot.PrintTree()
         print( self.data),
         if self.rightroot:
             self.rightroot.PrintTree()
    
def preoder(root: TreeNode):
    res = []
    def helper(root: TreeNode):
        if not root:
            return
        res.append(root.data)
        helper(root.leftroot)
        helper(root.rightroot)
    helper(root)
    return res

def preoder1(root: TreeNode):
    if not root:
        return []
    stack = [root]
    res = []
    while stack:
        cur = stack.pop()
        res.append(cur.data)
        if cur.rightroot:
            stack.append(cur.rightroot)
        if cur.leftroot:
            stack.append(cur.leftroot)
    return res

def midoder(root: TreeNode):
    res = []
    def helper(root):
        if not root:
            return
        helper(root.leftroot)
        res.append(root.data)
        helper(root.rightroot)
    helper(root)
    return res

def midoder1(root: TreeNode):
    if not root:
        return []
    cur = root
    res = []
    stack = []
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.leftroot
        cur = stack.pop()
        res.append(cur.data)
        cur = cur.rightroot
    return res

def postoder(root: TreeNode):
    res = []
    def helper(root):
        if not root:
            return
        helper(root.leftroot)
        helper(root.rightroot)
        res.append(root.data)
    helper(root)
    return res

def postoder1(root: TreeNode):
    if not root:
        return []
    stack = [root]
    res = []
    while stack:
        cur = stack.pop()
        res.append(cur.data)
        if cur.leftroot:
            stack.append(cur.leftroot)
        if cur.rightroot:
            stack.append(cur.rightroot)
    return res[::-1]

def BFS(root: TreeNode):
    if not root:
        return []
    queue = [root]
    res = []
    while queue:
        for _ in range(len(queue)):
            cur = queue.pop(0)
            if cur:
                res.append(cur.data)
                if cur.leftroot:
                    queue.append(cur.leftroot)
                if cur.rightroot:    
                    queue.append(cur.rightroot)
    return res



root = TreeNode(4)
root.insert(5)
root.insert(7)
root.insert(2)
root.insert(3)
root.insert(1)
print(midoder1(root))
