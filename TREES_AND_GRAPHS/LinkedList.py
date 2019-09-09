class Node:
    
    def __init__(self, value, next = None):
        self.value = value
        self.next = next

    def __str__(self):
        return str(self.value)

class LinkedList:

    def __init__(self, values_list = None):
        self.head = None
        self.tail = None
        for value in values_list:
            self.add_in_end(value)

    def add_in_end(self, value):
        if self.head is None:
            self.tail = self.head = Node(value)
        else:
            self.tail.next = Node(value)
            self.tail = self.tail.next

    def add_in_start(self, value):
        if self.head is None:
            self.tail = self.head = Node(value)    
        else:
            self.head = Node(value, self.head)

    def __iter__(self):
        current = self.head
        while current:
            yield current.value
            current = current.next

    def __str__(self):
        return ' -> '.join([str(x) for x in self])

    def __len__(self):
        result = 0
        current = self.head
        while current:
            result += 1
            current = current.next
        return result

