class Heap():
    def __init__(self) -> None:
        self.heaplist = [0]
        self.size = 0
    
    def insert(self, items):
        self.size += 1
        self.heaplist.append(items)
        self.up(self.size)

    def up(self, N):
        while N // 2 > 0:
            if self.heaplist[N // 2] > self.heaplist[N]:
                self.heaplist[N // 2], self.heaplist[N] = self.heaplist[N], self.heaplist[N // 2]
            N = N // 2

    def Minchild(self, N):
        if 2 * N + 1 > self.size:
            return 2 * N
        else:
            if self.heaplist[2 * N] < self.heaplist[2 * N + 1]:
                return 2 * N
            else:
                return 2 * N + 1
    
    def down(self, N):
        while 2 * N <= self.size:
            idx = self.Minchild(N)
            if self.heaplist[idx] < self.heaplist[N]:
                self.heaplist[idx], self.heaplist[N] = self.heaplist[N], self.heaplist[idx]
            N = idx
    
    def delMin(self):
        Min = self.heaplist[1]
        self.heaplist[1] = self.heaplist.pop()
        self.size -= 1
        self.down(1)
        return Min
    
    def buildheap(self, list):
        self.heaplist = self.heaplist + list[:]
        self.size = len(list)
        i = len(list) // 2
        while i > 0:
            self.down(i)
            i = i - 1 


    
if __name__ == '__main__':
    heap = Heap()
    heap.buildheap([5,6,9,2,1,7,3,8,4])
    reslut = heap.delMin()
    print(reslut)
    reslut = heap.delMin()
    print(reslut)
    reslut = heap.delMin()
    print(reslut)
    reslut = heap.delMin()
    print(reslut)
    reslut = heap.delMin()
    print(reslut)
    





    



