def bubble_sort(l):
    for _ in range(1, len(l)):
        for j in range(1, len(l)):
            if l[j] < l[j - 1]:
                temp = l[j - 1]
                l[j - 1] = l[j]
                l[j] = temp
    print(l)
# bubble_sort([4,3,5,6,2,3,1,7,8,15])

def selection_sort(l):
    for i in range(len(l)):
        min = l[i]
        for j in range(i + 1, len(l)):
            if l[j] < min: 
                min = l[j]
                l[j] = l[i]
                l[i] = min
    print(l)
# selection_sort([4,3,2,1,6,8,7,3,90,12,4,0])

def sort(l):
    helper = [None for x in l]
    merge_sort(l, helper, 0, len(l) - 1)
def merge_sort(l, helper, low, high):
    if low < high:
        middle = (low + high) // 2
        merge_sort(l, helper, low, middle)
        merge_sort(l, helper, middle + 1, high)
        sorted_merge(l, helper, low, middle, high)
def sorted_merge(l, helper, low, middle, high):
    helper = [x for x in l]
    hleft = low
    hright = middle + 1
    current = low
    while (hleft <= middle and hright <= high):
        if helper[hleft] < helper[hright]:
            l[current] = helper[hleft]
            hleft += 1
        else:
            l[current] = helper[hright]
            hright += 1
        current += 1
    remaining = middle - hleft + 1
    for r in range(remaining):
        l[current + r] = helper[hleft + r]
# l = [9,8,7,6,5,4,3,2,1]
# sort(l)
# print(l)

def search(l, x):
    return binary_search(l, x, 0, len(l))
def binary_search(l, x, low, high):
    if low >= high: return -1
    middle = (low + high) // 2
    if l[middle] == x:
        return middle
    elif l[middle] > x:
        return binary_search(l, x, low, middle)
    else:
        return binary_search(l, x, middle + 1, high)
print(search([0,1,2,3,4,5,6,7,8], 8))
    
    
    
        

            

             
            

