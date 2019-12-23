from LinkedList import LinkedList, Node

def get_middle(node):
    if node is None: return node
    slower = node
    faster = node.next
    while faster and faster.next:
        slower = slower.next
        faster = faster.next
    return slower

def sorted_merge(node1, node2):
    if node1 is None: return node2
    if node2 is None: return node1

    result = None
    if (node1.value <= node2.value):
        result = node1
        result.next = sorted_merge(node1.next, node2)
    else:
        result = node2
        result.next = sorted_merge(node1, node2.next)
    return result 

def merge_sort(h):
    if (h is None or h.next is None): return h
    
    middle = get_middle(h)
    next_middle = middle.next
    middle.next = None

    left = merge_sort(h)
    right = merge_sort(next_middle)

    return sorted_merge(left, right)

ll = LinkedList([4,1,3,6,8,7,1,2,3,5,4,2])

node = merge_sort(ll.head)

while node:
    print('{} -> '.format(node.value), end = ' ')
    node = node.next



