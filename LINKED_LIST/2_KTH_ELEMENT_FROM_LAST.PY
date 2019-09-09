from LinkedList import LinkedList

def get_kth_element_from_last(ll, k):
    current = ll.head
    runner = current

    for _ in range(k):
        if runner is None:
            return None
        runner = runner.next

    while runner:
        current = current.next
        runner = runner.next

    return current

ll = LinkedList([1,2,3,4,5,6,7,8])
print(get_kth_element_from_last(ll,1))
        
