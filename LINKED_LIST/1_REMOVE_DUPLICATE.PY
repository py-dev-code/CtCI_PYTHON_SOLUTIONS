from LinkedList import LinkedList

def remove_dups(ll):
    if ll.head is None:
        return

    current = ll.head
    uniq = set([current.value])
    while current.next:
        if current.next.value in uniq:
            current.next = current.next.next
        else:
            uniq.add(current.next.value)
            current = current.next
    return ll

def remove_dups_without_set(ll):
    if ll.head is None:
        return

    current = ll.head
    while current:
        runner = current
        while runner.next:
            if runner.next.value == current.value:
                runner.next = runner.next.next
            else:
                runner = runner.next
        current = current.next
    return ll


l1 = LinkedList([1,1,2,3,3,4,5,6,6])
print(remove_dups(l1))

l2 = LinkedList([1,1,2,3,3,4,5,6,6])
print(remove_dups_without_set(l2))