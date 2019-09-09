from LinkedList import LinkedList
from LinkedList import Node

def delete_middle_node(ll, node):
    print(node)
    # Trick is to copy the value of next node in current node and eventually get rid of the next node.
    node.value = node.next.value
    node.next = node.next.next

    return ll

ll = LinkedList([1,2,3,4])
print(ll)
print(delete_middle_node(ll, ll.head.next.next))



