class Node(object):
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
    def __repr__(self):
        return str(self.data)

class LinkedList(object):
    def __init__(self):
        self.head = self.tail = None
    def add_in_start(self, value):
        if self.head is None:
            self.head = self.tail = Node(value)
        else:
            self.head = Node(value, self.head)
    def add_in_end(self, value):
        if self.head is None:
            self.head = self.tail = Node(value)
        else:
            temp = self.tail
            self.tail = Node(value)
            temp.next = self.tail
    def __repr__(self):
        result = []
        node = self.head        
        while node:
            result.append(str(node.data))
            node = node.next
        return '->'.join(result)