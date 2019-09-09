from LinkedList import LinkedList, Node

def intersection(ll1, ll2):
    # Logic: Tails of both the lists should be same otherwise lists are not intersecting.
    # We will advance the longer list up to the difference of lengths.
    # Then we will iterate through the lists until we find the common node. Once we find it, rest of the list will be the intersection.
    if ll1.tail != ll2.tail:
        return None
    
    if len(ll1) <= len(ll2):
        shorter = ll1.head
        longer = ll2.head
    else:
        shorter = ll2.head
        longer = ll1.head

    for _ in range(abs(len(ll1) - len(ll2))):
        longer = longer.next

    while shorter:
        if shorter == longer:
            return shorter        
        else:
            shorter = shorter.next
            longer = longer.next

ll1 = LinkedList([1,2,3,4,5])
ll2 = LinkedList([5,6,7,8])
ll2.head.next = ll1.head.next.next
ll2.tail = ll1.tail

node = intersection(ll1, ll2)

if node:
    while node:
        print('{} ->'.format(node.value), end = " ")
        node = node.next
else:
    print("No Intersection")


        