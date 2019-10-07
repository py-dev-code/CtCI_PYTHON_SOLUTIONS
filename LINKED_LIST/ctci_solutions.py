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


# Probelm:2.1 Remove Dups: Write code to remove duplicates from an unsorted linked list.
# FOLLOW UP
# How would you solve this problem if a temporary buffer is not allowed?
def remove_dups_with_buffer(ll):
    '''
    If we can use an additional data structure then this can be done in linear time.
    Algorithm:        
        Create an empty list and add head in this list.
        Take head into a variable current. 
        Run the loop while current.next is not None. 
        If current.next.data is in the list then move current.next to current.next.next else 
        add current.next.data in the list and move current to current.next.
    '''
    if ll is None or ll.head is None:
        return

    buffer = []
    current = ll.head
    buffer.append(current.data)
    
    while current.next:
        if current.next.data in buffer:
            current.next = current.next.next
        else:
            buffer.append(current.next.data)
            current = current.next

def remove_dups_without_buffer(ll):
    '''
    If additional storage is not allowed then this will be done in quadratic time as we will have to compare current node
    with all the nodes in the list.
    Algorithm:
        take head into a variable current.
        Run the loop while current is not None. Inside the loop, create a runner node starting with current.
        Run 1 more loop while runner.next is not None. 
        If runner.next.data is equal to current.data then set runner.next to runner.next.next 
        else set runner to runner.next.
        Set current to current.next outside the inner loop.
    '''
    if ll is None or ll.head is None:
        return

    current = ll.head
    while current:
        runner = current
        while runner.next:
            if runner.next.data == current.data:
                runner.next = runner.next.next
            else:
                runner = runner.next
        current = current.next


# Probelm:2.2 Return Kth to Last: Implement an algorithm to find the kth to last element of a singly linked list.
def get_kth_to_last_node(ll, k):
    '''
    Algorithm:
        First, we will traverse k steps in the linked list and take kth node in runner node.
        After kth node is fetched, take head into a variable current.
        Run a loop while runner is not None. Move both current and runner by 1 step.        
        So when runner hits the end, current will be exactly k step behind from runner and that will be 
        the kth from last element in the list.
        This implement will return None if k is 0. If k is 1 then tail will be returned.
    Naive approach:
        If we are allowed to calculate the list length then kth from last element will be (length - k)th element
        from the beginning.
    '''
    if ll is None or ll.head is None or k <= 0:
        return

    runner = ll.head
    counter = 1
    counter_match = False
    while runner:
        if counter == k:
            counter_match = True
            break  
        runner = runner.next
        counter += 1

    if not counter_match:
        return None

    current = ll.head
    while runner.next:
        current = current.next
        runner = runner.next

    return current
    

# Probelm:2.3 Delete Middle Node: Implement an algorithm to delete a node in the middle (i.e., any node but
# the first and last node, not necessarily the exact middle) of a singly linked list, given only access to
# that node.
# EXAMPLE
# lnput:the node c from the linked list a->b->c->d->e->f
# Result: nothing is returned, but the new linked list looks like a->b->d->e->f
def delete_middle_node(ll, node):
    '''
    Algorithm:
        Assumption: Node can not be head or tail of the list and we have access to the node.
        With given information, we can delete the given node in constant time.
        Trick is to copy the next node data into given node and then get rid of the next node.
    '''
    if ll is None or ll.head is None or node == ll.head or node == ll.tail:
        return None

    node.data = node.next.data
    node.next = node.next.next


# Probelm:2.4 Partition: Write code to partition a linked list around a value x, such that all nodes less than x come
# before all nodes greater than or equal to x. If x is contained within the list, the values of x only need
# to be after the elements less than x (see below). The partition element x can appear anywhere in the
# "right partition"; it does not need to appear between the left and right partitions.
# EXAMPLE
# Input:
# Output:
# 3 -> 5 -> 8 -> 5 -> 10 -> 2 -> 1 [partition= 5]
# 3 -> 1 -> 2 -> 10 -> 5 -> 5 -> 8
def partition_list(ll, value):
    '''
    Algorithm:
        This is bit tricky.
        We will move through the list and if current node data is less than given value then move current node to the head
        else move it to tail.
        How to avoid infinite loop: reset the list tail to head before starting the loop.
        After the loop, check if tail.next is None or not. If not None then set it to None.
    '''
    if ll is None or ll.head is None:
        return

    ll.tail = ll.head
    current = ll.head

    while current:
        next = current.next
        current.next = None
        if current.data < value:
            current.next = ll.head
            ll.head = current
        else:
            ll.tail.next = current
            ll.tail = current
        current = next
    
    if ll.tail.next is not None:
        ll.tail.next = None
   

# Probelm:2.5 Sum Lists: You have two numbers represented by a linked list, where each node contains a single
# digit. The digits are stored in reverse order, such that the 1's digit is at the head of the list. Write a
# function that adds the two numbers and returns the sum as a linked list.
# EXAMPLE
# Input: (7-> 1 -> 6) + (5 -> 9 -> 2). That is, 617 + 295.
# Output: 2 -> 1 -> 9. That is, 912.
# FOLLOW UP
# Suppose the digits are stored in forward order. Repeat the above problem.
# EXAMPLE
# lnput:(6 -> 1 -> 7) + (2 -> 9 -> 5).That is,617 + 295.
# Output: 9 - > 1 -> 2. That is, 912.
def list_sum_reverse_order(ll1, ll2):
    '''
    Algorithm:
        If list is stored in reversed order then we just need to add the nodes starting from left and add the 
        result in end of a new list. We need to take care of carry and none values of the input list.
    '''
    if ll1 is None or ll2 is None or ll1.head is None or ll2.head is None:
        return

    result = LinkedList()
    node1 = ll1.head
    node2 = ll2.head
    carry = 0

    while node1 or node2:
        val1 = node1.data if node1 else 0
        val2 = node2.data if node2 else 0
        val = (val1 + val2 + carry) % 10
        carry = (val1 + val2 + carry) // 10
        result.add_in_end(val)
        if node1:
            node1 = node1.next
        if node2:
            node2 = node2.next

    if carry > 0:
        result.add_in_end(carry)

    return result

def list_sum_forward_order(ll1, ll2):
    '''
    Algorithm:
        If numbers are stored in forward order then we will pad the smaller list with 0's first.
        Then we will loop through the lists and calculate the total sum.
        In the end, we will convert the sum to a string and add each number to a new Linked List.
    '''
    if ll1 is None or ll2 is None or ll1.head is None or ll2.head is None:
        return

    node1 = ll1.head
    node2 = ll2.head

    n1 = 0
    n2 = 0
    while node1:
        n1 += 1
        node1 = node1.next
    while node2:
        n2 += 1
        node2 = node2.next
    
    if n1 > n2:
        for r in range(n1 - n2):
            ll2.add_in_start(0)
    elif n2 > n1:
        for r in range(n2 - n1):
            ll1.add_in_start(0)

    result = 0
    node1 = ll1.head
    node2 = ll2.head

    while node1 and node2:
        result = (result * 10) + node1.data + node2.data
        node1 = node1.next
        node2 = node2.next

    ll = LinkedList()    
    for r in [int(x) for x in str(result)]:
        ll.add_in_end(r)

    return ll
    

# Probelm:2.6 Palindrome: Implement a function to check if a linked list is a palindrome.
def is_palindrome(ll):
    if ll is None or ll.head is None:
        return

    current = ll.head
    runner = ll.head
    l = []

    while runner and runner.next:
        l.append(current.data)
        current = current.next
        runner = runner.next.next

    if runner is None:        
        node = current
    else:
        node = current.next

    while node:
        if node.data != l.pop():
            return False
        node = node.next

    return True
    

# Probelm:2.7 Intersection: Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting
# node. Note that the intersection is defined based on reference, not value. That is, if the kth
# node of the first linked list is the exact same node (by reference) as the jth node of the second
# linked list, then they are intersecting.
def intersection_node(ll1, ll2):
    '''
    Algorithm:
        If 2 linked lists are intersecting then they will have same nodes starting from the intersecting node.
        To traverse the lists, we will pad the smaller list with some dummy nodes.
        Once the length of both lists are equal, we just need to traverse the lists till we get the first
        matching node. The matching node will be the intersection point.
    '''
    if ll1 is None or ll2 is None or ll1.head is None or ll2.head is None:
        return

    n1 = n2 = 0
    node = ll1.head

    while node:
        n1 += 1
        node = node.next
    node = ll2.head
    while node:
        n2 += 1
        node = node.next

    if n1 > n2:
        for r in range(n1 - n2):
            ll2.add_in_start(0)
    elif n2 > n1:
        for r in range(n2 - n1):
            ll1.add_in_start(0)

    node1 = ll1.head
    node2 = ll2.head

    while node1 and node2:
        if node1 == node2:
            return node1
        else:
            node1 = node1.next
            node2 = node2.next

    return None


# Probelm:2.8 Loop Detection: Given a circular linked list, implement an algorithm that returns the node at the
# beginning of the loop.
# DEFINITION
# Circular linked list: A (corrupt) linked list in which a node's next pointer points to an earlier node, so
# as to make a loop in the linked list.
# EXAMPLE
# Input: A -> B -> C - > D -> E -> C [the same C as earlier]
# Output: C
def detect_cycle(ll):
    '''
    Algorithm:
        If linked list has a cycle then running faster and slower traversal will collide for some node.
        Once the 2 pointers collide, reset slower pointer to head.
        Traverse both the slower and faster pointer by 1 step till they collide. Collission point will be
        the cycle node.
    '''
    if ll is None or ll.head is None:
        return

    current = ll.head
    runner = ll.head
    pointer_matched =  False

    while runner.next:
        current = current.next
        runner = runner.next.next
        if current == runner:
            current = ll.head
            pointer_matched = True
            break    

    if not pointer_matched:
        return None            

    while current:
        if current == runner:
            return current
        else:
            current = current.next
            runner = runner.next


if  __name__ == "__main__":
    print("Problem# 2.1")
    ll1 = LinkedList()
    ll2 = LinkedList()
    for r in [1,2,3,4,4,3,5,6,7,1,2]:
        ll1.add_in_end(r)  
        ll2.add_in_end(r)  

    remove_dups_with_buffer(ll1)          
    print(ll1)
    remove_dups_without_buffer(ll2)
    print(ll2)

    print("\nProblem# 2.2")
    ll = LinkedList()
    for r in [1,2,3,4,5,6,7]:
        ll.add_in_end(r)
    print(get_kth_to_last_node(ll, 7))

    print("\nProblem# 2.3")
    ll = LinkedList()
    for r in [1,2,3,4]:
        ll.add_in_end(r)
    delete_middle_node(ll, ll.head.next)
    print(ll)

    print("\nProblem# 2.4")
    ll = LinkedList()
    for r in [3, 5, 8, 5, 10, 2, 1,1,2,3,4]:
        ll.add_in_end(r)
    print(ll)
    partition_list(ll, 5)
    print(ll)

    print("\nProblem# 2.5")
    ll1 = LinkedList()
    ll2 = LinkedList()
    for r in [7, 1, 6]:
        ll1.add_in_end(r)
    for r in [5, 9, 2]:
        ll2.add_in_end(r)
    print(list_sum_reverse_order(ll1, ll2))

    ll1 = LinkedList()
    ll2 = LinkedList()
    for r in [1,6]:
        ll1.add_in_end(r)
    for r in [9,6]:
        ll2.add_in_end(r)
    print(list_sum_forward_order(ll1, ll2))

    print("\nProblem# 2.6")
    ll = LinkedList()
    for r in [1,2,3,2,1]:
        ll.add_in_end(r)
    print(is_palindrome(ll))

    print("\nProblem# 2.7")
    ll1 = LinkedList()
    ll1.add_in_end(1)
    ll1.add_in_end(2)

    ll2 = LinkedList()
    ll2.add_in_end(-1)

    for r in [4,5,6]:
        node = Node(r)
        ll1.tail.next = node
        ll1.tail = node
        ll2.tail.next = node
        ll2.tail = node

    print(intersection_node(ll1, ll2))

    print("\nProblem# 2.8")
    ll = LinkedList()
    for r in [1,2,3,4,5,6,7,8,9]:
        ll.add_in_end(r)
    ll.tail.next = ll.head.next.next.next.next
    print(detect_cycle(ll))
