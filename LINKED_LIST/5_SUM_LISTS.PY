from LinkedList import LinkedList

def my_sum_from_head(ll1, ll2):
    # forward sum will be similar. We just don't need to reverse the string in the end
    l1 = []
    l2 = []

    current = ll1.head    
    while current:
        l1.append(str(current.value))
        current = current.next

    current = ll2.head    
    while current:
        l2.append(str(current.value))
        current = current.next   

    result = int(''.join([x for x in reversed(l1)])) + int(''.join([x for x in reversed(l2)]))

    # reverse this list again to create the output LinkedList
    ll = LinkedList([int(x) for x in str(result)[::-1]])

    return ll

def sum_from_head(ll1, ll2):
    n1 = ll1.head
    n2 = ll2.head
    lst = []
    carry = 0
    while n1 or n2:
        result = carry
        if n1:
            result += n1.value
            n1 = n1.next
        if n2:
            result += n2.value
            n2 = n2.next
        
        lst.append(result % 10)
        carry = result // 10

    if carry:
        lst.append(carry) 

    return LinkedList(lst)

def sum_from_tail(ll1, ll2):
    # padding the shorter list with 0s
    if (len(ll1) < len(ll2)):
        for _ in range(len(ll2) - len(ll1)):
            ll1.add_in_start(0)
    elif (len(ll1) > len(ll2)):
        for _ in range(len(ll1) - len(ll2)):
            ll2.add_in_start(0)        

    n1 = ll1.head
    n2 = ll2.head
    result = 0
    while n1 and n2:
        result = (result * 10) + n1.value + n2.value
        n1 = n1.next
        n2 = n2.next

    return LinkedList([int(x) for x in str(result)])                

ll1 = LinkedList([3,2,1])
ll2 = LinkedList([3,2,1])
print(my_sum_from_head(ll1, ll2))
print(sum_from_head(ll1, ll2))
print(sum_from_tail(ll1, ll2))

    