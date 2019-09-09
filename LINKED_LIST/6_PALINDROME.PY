from LinkedList import LinkedList

def my_is_palindrome(ll):
    current = ll.head
    lst = []
    while current:
        lst.append(current.value)
        current = current.next

    return lst == lst[::-1]

def is_palindrome(ll):
    # make a List with first half elements. In case of 5, make till 2.
    # Move the pointer to 2nd half beginning and iterate through to compare the values by popping the values from list.
    fast = slow = ll.head
    lst = []
    while fast and fast.next:
        lst.append(slow.value)
        slow = slow.next
        fast = fast.next.next
    
    # This will happen in odd length cases
    if fast:
        slow = slow.next

    while slow:
        top = lst.pop()
        if slow.value != top:
            return False
        slow = slow.next

    return True

ll = LinkedList([1,2,3,3,2,1])
print(my_is_palindrome(ll))
print(is_palindrome(ll))