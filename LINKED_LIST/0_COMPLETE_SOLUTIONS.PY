class Node:
    def __init__(self, value, next = None):
        self.data = value
        self.next = next
    def __str__(self):
        return str(self.data)

class LinkedList:
    def __init__(self, values = None):
        self.head = None
        self.tail = None
        if values is not None:
            for r in values:
                self.add_in_end(r)    
    def add_in_end(self, value):
        if self.head is None:
            self.head = self.tail = Node(value)
        else:
            self.tail.next = Node(value)
            self.tail = self.tail.next
    def add_in_start(self, value):
        if self.head is None:
            self.head = self.tail = Node(value)
        else:
            self.head = Node(value, self.head)
    def __iter__(self):
        node = self.head
        while node:
            yield node.data
            node = node.next
    def __str__(self):
        values = [str(x) for x in self]
        return ' -> '.join(values)
    def __len__(self):
        result = 0
        node = self.head
        while node:
            result += 1
            node = node.next
        return result

def get_middle(node):
    slower = node
    faster = node.next
    while faster:
        faster = faster.next
        if faster:
            slower = slower.next
            faster = faster.next
    return slower    

def merge_sort(node):
    if node is None or node.next is None: return node
    middle = get_middle(node)
    middle_next = middle.next
    middle.next = None
    left = merge_sort(node)
    right = merge_sort(middle_next)
    return sorted_merge(left, right)
def sorted_merge(node1, node2):
    if node1 is None: return node2
    if node2 is None: return node1
    if (node1.data <= node2.data):
        result = node1
        result.next = sorted_merge(node1.next, node2)
    else:
        result = node2
        result.next = sorted_merge(node1, node2.next)    
    return result   
# ll = LinkedList([8,9,7,1,2,5,6,4,3])
# node = merge_sort(ll.head)
# while node:
#     print(node.data, end = ' ')    
#     node = node.next

def get_loop_node(ll):
    slower = ll.head
    faster = ll.head
    while (faster is not None and faster.next is not None):
        slower = slower.next        
        faster = faster.next.next
        if (slower == faster):
            break
    if faster is None or faster.next is None: return "No Loop Detected"        
    slower = ll.head
    while (slower != faster):
        slower = slower.next
        faster = faster.next
    return slower    

def get_intersection_node(ll1, ll2):
    if ll1.tail != ll2.tail: return "Lists are not intersecting"
    if (len(ll1) >= len(ll2)): 
        longer = ll1.head
        shorter = ll2.head
    else: 
        longer = ll2.head
        shorter = ll1.head
    for _ in range(abs(len(ll1) - len(ll2))):
        longer = longer.next
    while shorter:
        if longer == shorter:
            return longer
        else:
            longer = longer.next
            shorter = shorter.next

def is_palindrome_mine(ll):
    return [x for x in ll] == [x for x in ll][::-1]

def is_palindrome_book(ll):
    slow = ll.head
    fast = ll.head
    half_list = []
    while fast is not None and fast.next is not None:
        half_list.append(slow.data)
        slow = slow.next
        fast = fast.next.next
    if fast is not None:
        slow = slow.next
    while slow:
        last_from_list = half_list.pop()
        if last_from_list != slow.data:
            return False
        slow = slow.next    
    return True    

def sum_forward_mine(ll1, ll2): # 123 + 123 = 246
    l1 = [str(x) for x in ll1]
    l2 = [str(x) for x in ll2]
    result = int(''.join(l1)) + int(''.join(l2))
    values = [x for x in list(str(result))]
    return LinkedList(values)
def sum_backward_mine(ll1, ll2):    #321 + 321 = 642
    l1 = [str(x) for x in ll1][::-1]
    l2 = [str(x) for x in ll2][::-1]
    result = int(''.join(l1)) + int(''.join(l2))
    values = [x for x in list(str(result))][::-1]
    return LinkedList(values)

def sum_forward_book(ll1, ll2):
    if len(ll1) <= len(ll2):
        for _ in range(len(ll2) - len(ll1)):
            ll1.add_in_start(0)
    else:
        for _ in range(len(ll1) - len(ll2)):
            ll2.add_in_start(0)
    result = 0
    n1 = ll1.head
    n2 = ll2.head
    while n1 and n2:
        result = (result * 10) + n1.data + n2.data
        n1 = n1.next
        n2 = n2.next
    return LinkedList([int(x) for x in str(result)])
def sum_backward_book(ll1, ll2):
    n1 = ll1.head
    n2 = ll2.head
    carry = 0
    ll = LinkedList(None)
    while n1 or n2:
        result = carry
        if n1:
            result = result + n1.data
            n1 = n1.next
        if n2:
            result = result + n2.data
            n2 = n2.next
        carry = result // 10
        ll.add_in_end(result % 10)                
    if carry > 0: ll.add_in_end(carry)
    return ll

def list_partition(ll, value):
    result = LinkedList(None)
    node = ll.head
    while node:
        if node.data < value:
            result.add_in_start(node.data)
        else:
            result.add_in_end(node.data)
        node = node.next
    return result

def delete_middle_node(ll, node):
    node.data = node.next.data
    node.next = node.next.next
    return ll

def return_kth_node_from_last(ll, k):
    slower = ll.head
    faster = ll.head
    for _ in range(k):
        faster = faster.next
    while faster:
        faster = faster.next
        slower = slower.next
    return slower

def remove_duplicate_with_set(ll):
    if ll.head is None or ll.head.next is None: return ll
    uniq = set([ll.head.data])
    node = ll.head
    while node.next:
        if node.next.data in uniq:
            node.next = node.next.next
        else:
            uniq.add(node.next.data)    
            node = node.next
    return ll

def remove_duplicate_without_set(ll):
    if ll.head is None or ll.head.next is None: return ll
    current = ll.head
    while current:
        runner = current
        while runner.next:
            if (runner.next.data == current.data):
                runner.next = runner.next.next
            else:
                runner = runner.next
        current = current.next
    return ll
        
# ll = LinkedList([1,5,3,3,7,2,0,9,3])
# print(remove_duplicate_with_set(ll))
# print(remove_duplicate_without_set(ll))
        
# ll1 = LinkedList([1,2,3,4,5])
# ll2 = LinkedList([5,6,7,8])
# ll2.head.next = ll1.head.next.next
# ll2.tail = ll1.tail
# print(get_intersection_node(ll1, ll2))

# ll.tail.next = ll.head.next.next
# print(get_loop_node(ll))
