from LinkedList import LinkedList, Node

def get_loop_node(ll):
    
    slow = ll.head
    fast = ll.head

    # Running slow and fast pointers until they collide. 
    # They will collide at k distance from start of the head and k steps before loop head.
    while (fast is not None and fast.next is not None):
        slow = slow.next
        fast = fast.next.next
        if (slow == fast):
            break

    # If fast or fast.next is None then LinkedList is not having any Loop
    if fast is None or fast.next is None:
        return "No Loop Detected"

    # Move slow to head and keep fast as is.
    # Move them 1 step at a time unless they collide.
    # Collision point is the loop Node
    slow = ll.head
    while (slow != fast):
        slow = slow.next
        fast = fast.next

    return slow

ll = LinkedList([1,2,3,4,5,6,7,8]) 
ll.tail.next = ll.head.next.next.next  # Putting Loop Node at 4
print(get_loop_node(ll))

