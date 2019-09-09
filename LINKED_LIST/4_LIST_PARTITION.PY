from LinkedList import LinkedList, Node
from LinkedList import Node

def list_partition(ll, value):
    '''
    input will be a linked list. List will be partitioned on the given value in such a way that
    elements less than value will come before all elements having value more than or equal to given input.
    Approach: iterate through the list, if element is less than value: move it to the head else move it to the end.
    '''

    # making tail as head means to start a new linked list here. 
    # If we dont reset the tail then we will be working on the given LinkedList only which can go in an infinite loop.
    current = ll.tail = ll.head # tail value can still be retrieved
    while current:
        nextNode = current.next
        current.next = None
        if current.value < value:
            current.next = ll.head
            ll.head = current
        else:
            ll.tail.next = current
            ll.tail = current
        current = nextNode

    if ll.tail.next is not None:
        ll.tail.next = None

ll = LinkedList([2,1,3])
print(ll)
list_partition(ll, 10)
print(ll)












