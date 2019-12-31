class Stack(object):
    class StackNode(object):
        def __init__(self, data, next=None):
            self.data = data
            self.next = next
    def __init__(self):
        self.top = None
    def pop(self):
        if self.top is None:
            raise ValueError("Stack is Empty")
        else:
            item = self.top.data
            self.top = self.top.next
            return item
    def push(self, item):
        if self.top is None:
            self.top = Stack.StackNode(item)
        else:
            self.top = Stack.StackNode(item, self.top)
    def peek(self):
        if self.top is None:
            raise ValueError("Stock is Empty")
        else:
            return self.top.data
    def isEmpty(self):
        return self.top is None
    def __repr__(self):
        result = []
        node = self.top
        while node:
            result.append(str(node.data))
            node = node.next
        return ' -> '.join(result)

class Queue(object):
    class QueueNode(object):
        def __init__(self, data, next=None):
            self.data = data
            self.next = next
    def __init__(self):
        self.first = None
        self.last = None
    def add(self, item):
        if self.first is None:
            self.first = self.last = Queue.QueueNode(item)
        else:
            node = Queue.QueueNode(item)
            self.last.next = node
            self.last = node
    def remove(self):
        if self.first is None:
            raise ValueError("Queue is Empty")
        else:
            item = self.first.data
            self.first = self.first.next
            if self.first is None:
                self.first = self.last = None
            return item
    def peek(self):
        if self.first is None:
            raise ValueError("Queue is Empty")
        else:
            return self.first.data
    def isEmpty(self):
        return self.first is None
    def __repr__(self):
        result = []
        node = self.first
        while node:
            result.append(str(node.data))
            node = node.next
        return ' -> '.join(result)

# Problem: 3.1 Three in One: Describe how you could use a single array to implement three stacks.
'''
    Approach 1: Fixed Space Division Approach.
        1. Each stack will be allocated a fixed amount of space and it can only grow within this space.
        2. Attributes: number_of_stacks=3, array_capacity, data=[], stack_sizes=[]
           Methods:
                push(stack_num, item)
                pop(stack_num)
                peek(stack_num)
                isEmpty(stack_num)
    
    Approach 2: Variable Space Division Approach.
        1. In this approach, if one stack is running full then it can grow in space by moving elements from other stacks.
        2. This approach will be complicated.

    Implementations of these approaches will be given later.
'''

# Problem: 3.2 Stack Min: How would you design a stack which, in addition to push and pop, has a function min
# which returns the minimum element? Push, pop and min should all operate in 0(1) time.
'''
    Algorithm:
        1. In this, we cant just maintain a single attribute min_val in Stack class as when we pop the min
        element, we need to search the entire stack to find the new min. This will break the O(1) time 
        constraint.
        2. We can create the StackNode so that each element will keep the min when it was added so whenever the
        element is popped from the Stack, its local min will become the Stack Min.
        3. This algorithm is implemented below.
    Optimized Approach:
        1. This approach will double the Space needed to store the Stack.
        2. We can rather maintain a separate Stack to track the mins which will reduce the space needed.
'''
class StackWithMin(object):
    class StackWithMinNode(object):
        def __init__(self, data, least=None, next=None):
            self.data = data
            self.least = least
            self.next = next
    def __init__(self):
        self.top = None
    def pop(self):
        if self.top is None:
            raise ValueError("Stock is Empty")
        else:
            item = self.top.data
            self.top = self.top.next
            return item
    def push(self, item):
        if self.top is None:
            self.top = StackWithMin.StackWithMinNode(item, item)
        else:
            self.top = StackWithMin.StackWithMinNode(item, min(self.top.least, item), self.top)
    def peek(self):
        if self.top is None:
            raise ValueError("Stock is Empty")
        else:
            return self.top.data
    def least(self):
        if self.top is None:
            raise ValueError("Stock is Empty")
        else:
            return self.top.least
    def isEmpty(self):
        return self.top is None
    def __repr__(self):
        result = []
        node = self.top
        while node:
            result.append(str(node.data))
            node = node.next
        return ' -> '.join(result)


# Problem: 3.3 Stack of Plates: Imagine a (literal) stack of plates. If the stack gets too high, it might topple.
# Therefore, in real life, we would likely start a new stack when the previous stack exceeds some
# threshold. Implement a data structure SetOfStacks that mimics this. SetOfStacks should be
# composed of several stacks and should create a new stack once the previous one exceeds capacity.
# SetOfStacks.push() and SetOfStacks.pop() should behave identically to a single stack
# (that is, pop () should return the same values as it would if there were just a single stack).
# FOLLOW UP
# Implement a function popAt(int index) which performs a pop operation on a specific substack.
'''
    Algorithm:
        1. This is a big problem to code but not very complicated one.
        2. SetOfStacks class will have a list that will store the list of Stack.
        3. Stack class will need to have top, bottom, size and capacity attributes.
        3. Stack class need to implement both pop() and popBottom() methods.
        4. popAt() method can be implemented in 2 ways:
            1. Just pop the Stack at a given index and no shifting of elements. This will perform in O(1) time
            but it will give issues if program is assuming that all the Stacks are operating at Full capacity.
            2. We will pop the Stack at given index and rollover the elements from other Stacks. This will
            need removing the Bottom of next Stack and pushing it to the current Stack. This implementation needs
            little bit of precaution to handle list indexes and size. We have implemented popAt() with 2nd approach.
'''
class SetOfStacks(object):
    class Node(object):
        def __init__(self, data, next=None):
            self.data = data
            self.next = next
        def __repr__(self):
            return str(self.data)
    
    class Stack(object):
        def __init__(self, capacity=5):
            self.capacity = capacity
            self.size = 0
            self.top = self.bottom = None
        def push(self, item):
            if self.size == self.capacity:
                raise ValueError("Stack is Full")
            else:
                if self.top is None:
                    self.top = self.bottom = SetOfStacks.Node(item)
                else:
                    self.top = SetOfStacks.Node(item, self.top)
                self.size += 1
        def isEmpty(self):
            return self.top is None
        def pop(self):
            if self.isEmpty():
                raise ValueError("Stack is Empty")
            item = self.top.data
            self.top = self.top.next
            if self.top is None:
                self.top = self.bottom = None
            self.size -= 1
            return item
        def popBottom(self):
            if self.isEmpty():
                raise ValueError("Stack is Empty")
            item = self.bottom.data
            node = self.top
            if node == self.bottom:
                self.top = self.bottom = None
            else:
                while node:
                    if node.next == self.bottom:
                        node.next = None
                        self.bottom = node
                    node = node.next
            self.size -= 1
            return item            
        def __repr__(self):
            result = []
            node = self.top
            while node:
                result.append(str(node.data))
                node = node.next
            return ' -> '.join(result)
    
    def __init__(self, capacity=5):
        self.stacks = []
        self.capacity = capacity

    def isEmpty(self):
        return len(self.stacks) == 0
    
    def push(self, item):
        if len(self.stacks) == 0:
            self.stacks.append(SetOfStacks.Stack(self.capacity))
        try:
            self.stacks[len(self.stacks) - 1].push(item)
        except ValueError:
            self.stacks.append(SetOfStacks.Stack(self.capacity))
            self.stacks[len(self.stacks) - 1].push(item)            

    def pop(self):
        if self.isEmpty():
            raise ValueError("Stack is Empty")
        else:
            item = self.stacks[len(self.stacks) - 1].pop()
            if self.stacks[len(self.stacks) - 1].size == 0:
                self.stacks.remove(self.stacks[len(self.stacks) - 1])
            return item

    def popAt(self, stack_index):
        if stack_index > len(self.stacks):
            raise KeyError("Specified Stack_Index doesn't exist")
        stack = self.stacks[stack_index]
        item = stack.pop()
        n = len(self.stacks)
        for r in range(stack_index + 1, n):
            stack.push(self.stacks[r].popBottom())
            if self.stacks[r].size == 0:
                self.stacks.remove(self.stacks[r])
            # If last Stack has only 1 item and now it has been removed
            if r == len(self.stacks):
                break
            else:
                stack = self.stacks[r]
        return item

    def __repr__(self):
        result = []
        for r in range(len(self.stacks)):
            result.append(f'{r}: {str(self.stacks[r])}')
        return '\n'.join(result)


# Problem: 3.4 Queue via Stacks: Implement a MyQueue class which implements a queue using two stacks.
'''
    Algorithm:
        1. 2 Stacks will be there: S1 and S2. S1 will always be used for Push/Enqueue. S2 Stack will only be used for
        Pop/Dequeue.
        2. Enqueue: If S2 is Empty, push to S1. Else Move data from S2 to S1 and push to S1.
        3. Dequeue/Peek: If S2 is not Empty, pop from it. Else move data from S1 to S2 and Pop/Peek from it.
    
    Side Notes: Implementing a Stack with 2 Queues:
        1. 2 ways to implement: Making Push operation costly or making Pop operation costly. 2nd approach works as below:
        2. 2 Queues will be there: Q1 and Q2.
        3. Push: Always push to Q1.
        4. Pop: Pop everything from Q1 to Q2 while storing the last popped item. Once done, swap the names between Q1 and Q2.
           Return the last popped item.    
'''
class QueueWithTwoStacks(object):
    def __init__(self):
        self.first = Stack()
        self.second = Stack()

    def isEmpty(self):
        return self.first.isEmpty() and self.second.isEmpty()

    def move_data(self, stack1, stack2):
        while not stack1.isEmpty():
            stack2.push(stack1.pop())
    
    def add_item(self, item):
        if self.second.isEmpty():
            self.first.push(item)
        else:
            self.move_data(self.second, self.first)
            self.first.push(item)

    def remove_item(self):
        if self.isEmpty():
            raise ValueError("Queue is Empty")
        if not self.second.isEmpty():
            return self.second.pop()
        else:
            self.move_data(self.first, self.second)
            return self.second.pop()

    def peek(self):
        if self.isEmpty():
            raise ValueError("Queue is Empty")
        if not self.second.isEmpty():
            return self.second.peek()
        else:
            self.move_data(self.first, self.second)
            return self.second.peek()

    def __repr__(self):
        if not self.first.isEmpty():
            self.move_data(self.first, self.second)
        return str(self.second)


# Problem: 3.5 Sort Stack: Write a program to sort a stack such that the smallest items are on the top. You can use
# an additional temporary stack, but you may not copy the elements into any other data structure
# (such as an array). The stack supports the following operations: push, pop, peek, and is Empty.
'''
    Algorithm:
        1. We can only use 1 additional Stack.
        2. We need to maintain S2 always in Sorted Order.
        3. Pop from S1 and store it in temp. Peek from S2, if peek is more than temp, pop from S2 and push to S1. Repeat 
        till S2 is empty or peek is less than temp. Push the temp to S2. 
        4. Repeat number 3 till S1 is empty.
        5. Time Complexity: O(N^2)
'''
class StackWithSort(object):
    class StackNode(object):
        def __init__(self, data, next=None):
            self.data = data
            self.next = next
    def __init__(self):
        self.top = None

    def sort(self):
        if self.isEmpty():
            raise ValueError("Stack is Empty")
        s2 = Stack()
        while not self.isEmpty():
            s2.push(self.pop())

        while not s2.isEmpty():
            temp = s2.pop()
            if self.isEmpty():
                self.push(temp)
            else:
                while not self.isEmpty() and temp > self.peek():
                    s2.push(self.pop())
                self.push(temp)

    def pop(self):
        if self.top is None:
            raise ValueError("Stack is Empty")
        else:
            item = self.top.data
            self.top = self.top.next
            return item
    def push(self, item):
        if self.top is None:
            self.top = StackWithSort.StackNode(item)
        else:
            self.top = StackWithSort.StackNode(item, self.top)
    def peek(self):
        if self.top is None:
            raise ValueError("Stock is Empty")
        else:
            return self.top.data
    def isEmpty(self):
        return self.top is None
    def __repr__(self):
        result = []
        node = self.top
        while node:
            result.append(str(node.data))
            node = node.next
        return ' -> '.join(result)    
    

# Problem: 3.6 Animal Shelter: An animal shelter, which holds only dogs and cats, operates on a strictly "first in, first
# out" basis. People must adopt either the "oldest" (based on arrival time) of all animals at the shelter,
# or they can select whether they would prefer a dog or a cat (and will receive the oldest animal of
# that type). They cannot select which specific animal they would like. Create the data structures to
# maintain this system and implement operations such as enqueue, dequeueAny, dequeueDog,
# and dequeueCat. You may use the built-in Linkedlist data structure.
'''
    Algorithm:
        1. We will maintaint 2 Queues (Dogs and Cats) in a wrapper Animal Class.
        2. Whenever we add a Dog or Cat in its respective Queue, we will assign a Sequence Number so that
        when we apply dequeueAny(), it will pick the oldest animal based on this Sequence Number.
        3. Remaining all the operations will be Straight Forward.
'''
class AnimalShelter(object):
    def __init__(self):
        self.dogs = Queue()
        self.cats = Queue()
        self.sequence = 0
    
    def isEmpty(self):
        return self.dogs.isEmpty() and self.cats.isEmpty()

    def enqueue(self, animal_type, name):
        if animal_type is None or animal_type not in ['Dog', 'Cat'] or name is None:
            raise TypeError("Animal Type and Name are needed.")
        self.sequence += 1
        if animal_type == "Dog":
            self.dogs.add((name, self.sequence))
        else:
            self.cats.add((name, self.sequence))
        
    def dequeue(self, animal_type):
        if animal_type is None or animal_type not in ['Dog', 'Cat']:
            raise TypeError("Animal Type is needed.")
        if animal_type == "Dog":
            if self.dogs.isEmpty():
                raise ValueError("No Dogs are left.")
            else:
                return self.dogs.remove()
        else:
            if self.cats.isEmpty():
                raise ValueError("No Cats are left.")
            else:
                return self.cats.remove()

    def dequeueAny(self):
        if self.isEmpty():
            raise ValueError("No Animals are left.")
        if (( not self.dogs.isEmpty() and not self.cats.isEmpty() 
             and self.dogs.peek()[1] < self.cats.peek()[1] ) or (self.cats.isEmpty()) ):
            return self.dogs.remove()
        else:
            return self.cats.remove()


if __name__ == "__main__":
    print("Problem: 3.2")
    s = StackWithMin()
    s.push(3)
    s.push(4)
    s.push(2)
    s.push(1)
    print(s, ':', s.least())
    s.pop()
    print(s, ':', s.least())
    s.pop()
    print(s, ':', s.least())
    s.pop()
    print(s, ':', s.least())
    s.pop()
    print(s, ':', s.least())

    print("\nProblem: 3.3 SetOfStacks")
    ss = SetOfStacks()
    for r in range(15):
        ss.push(r)
    print(ss)
    print("Push to add a new Stack:")
    ss.push(15)
    print(ss)
    print("Pop to remove the last Stack:")
    print(ss.pop())
    print(ss)
    print("PopAt to remove the last Stack with Rollover:")
    ss.push(15)
    print(ss.popAt(0))
    print(ss)

    print("\nProblem: 3.4 Queue with 2 Stacks:")
    q = QueueWithTwoStacks()
    print("Adding Items:")
    for r in range(5):
        q.add_item(r)
    print(q)
    print("Removing Items:")
    print(q.remove_item())
    print(q.remove_item())
    print(q)
    print("Adding Items after Remove operation:")
    q.add_item(1)
    q.add_item(2)
    print(q)
    print("Peeking Operation:")
    print(q.peek())
    print(q)

    print("\nProblem: 3.5 Stack Sorting with an Additional Stack:")
    s = StackWithSort()
    for r in [7,6,3,4,5,7,1,1,2,2,34]:
        s.push(r)
    print("Before Sort:")
    print(s)
    print("After Sort:")
    s.sort()
    print(s)

    print("\nProblem: 3.6 Animal Shelter:")
    a = AnimalShelter()
    a.enqueue('Dog', 'Dog1')
    a.enqueue('Cat', 'Cat1')
    a.enqueue('Cat', 'Cat2')
    a.enqueue('Dog', 'Dog2')
    print(a.dogs, '::', a.cats)
    print(a.dequeue('Dog'))
    print(a.dequeueAny())
    print(a.dequeueAny())
    print(a.dequeueAny())
    print(a.dequeueAny())




