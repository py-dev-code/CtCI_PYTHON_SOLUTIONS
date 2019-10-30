'''
    This script illustrates the implementation of all the basic sorting algorithms.
    1. merge_sort.
    2. Quick_sort.
    3. Heap_sort.
    4. Bubble/Insertion sort.
    5. Selection sort.
'''

def merge_sort(arr):
    '''
    Merge Sort is a very basic and efficient Sorting algorithm which takes O(NlogN) run time.
    Algorithm:
        We will sort the array by dividing it into 2 halves and sort each half individually and later merge the sorted halves.
        To do this, we need to use a helper array.
    merge_sort_util function:
        1. define a merge_sort_util function that will take array, helper, low and high.
        2. Base condition for this function will be when low <= high.
        3. mid = (low + high) // 2, call same function for left half (low is low and high is mid) and right half (low is mid + 1
        and high is high).
        4. Once both the calls are done, we should have both the halves sorted.
        5. In the end, call merge function with inputs as arr, helper, low, mid and high as inputs.
    merge function:
        1. merge function will sort the input array with the help of helper array. It will merge the 2 halves of the array (defined 
        by low, mid and high values) which are already sorted.
        1. copy everything from arr into helper array.
        2. We will maintain 3 pointers: left, right and current starting from low, mid + 1 and low respectively.
        3. run a loop till left <= mid and right <= high and inside the loop check if helper[left] < helper[right],
        based on the comparison, set arr[current] and increase all 3 pointers accordingly.
        4. After the loop, if left array has anything left then we need to copy those elements from helper and set them into array.
        5. After the loop, if right array has anything left then no need to do anything as they will already be in sorted order.
    '''
    def merge_sort_util(arr, helper, low, high):
        if low >= high:
            return
        mid = (low + high) // 2
        merge_sort_util(arr, helper, low, mid)
        merge_sort_util(arr, helper, mid + 1, high)
        merge(arr, helper, low, mid, high)

    def merge(arr, helper, low, mid, high):
        helper = list(arr)
        left = low
        right = mid + 1
        current = low
        while left <= mid and right <= high:
            if helper[left] <= helper[right]:
                arr[current] = helper[left]
                left += 1
            else:
                arr[current] = helper[right] 
                right += 1
            current += 1
        
        remaining = mid - left
        for r in range(remaining + 1):
            arr[current + r] = helper[left + r]        

    if arr is None or len(arr) == 0:
        return
    helper = []
    merge_sort_util(arr, helper, 0, len(arr) - 1)
    return arr

def quick_sort(arr):
    '''
    Quick Sort is another efficient sorting algorithm which sorts an array in O(NlogN) time.
    Algorithm:
        Quick sort is done by selecting a random point in an array and then swapping the elements from left and right half
        so that all left elements will be smaller than Pivot and all right elements will be larger than pivot.
        We can implement the above logic by creating 3 util functions.
    1. swap_elements function: this will just take 2 indexes and swap their values in given array.
    2. partition function: this will take array, low and high as inputs.
        First, we will select a pivot value by selecting a random point in the array.
        We will run a loop till low <= high.
            First, we will adjust the low pointer till we find a point that is larger than pivot point by increasing it with 1.
            Then, we will adjust the high pointer till we find a point that is smaller than pivot point by decreasing it with 1.
            if low <= high then we will swap the low and high elements. Increment low by 1 and Decrement high by 1.
            Repeat the loop.
        In the end return low.
        So, if we have got low as 0, high as 4 and pivot has been selected as mid point (that is 2) then partition will return 3 
        as a result.
    3. quick_sort_util function: This function will also take array, low and high as inputs. This will be the starting point of 
        the algorithm and will be called recursively.
        First, we will get an index point by calling partition function in the array.
        If low < index - 1 then call quick_sort_util for low to index - 1.
        If index < high then call quick_sort_util for index to high.
        In example take in point 2, we got index value as 3.
        So, next it will apply quick_sort on 0 to 2 and 3 to 4. Once these 2 parts are sorted, whole array will be sorted 
        completely.
    '''
    def swap_elements(arr, i, j):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp

    def partition(arr, low, high):
        pivot = arr[(low + high) // 2]
        
        while low <= high:

            while arr[low] < pivot:
                low += 1
            while arr[high] > pivot:
                high -= 1
            if low <= high:
                swap_elements(arr, low, high)
                low += 1
                high -= 1
        
        return low

    def quick_sort_util(arr, low, high):
        index = partition(arr, low, high)
        if low < index - 1:
            quick_sort_util(arr, low, index - 1)
        if index < high:
            quick_sort_util(arr, index, high)

    if arr is None or len(arr) == 0:
        return
    quick_sort_util(arr, 0, len(arr) - 1)
    return arr


def heap_sort(arr):
    '''
    Heap sort is one more sorting algorithm that can complete the sorting in O(NlogN) time.
    Algorithm:
        1. Make a max binary heap with the given array.
        2. Swap the root of binary heap with its last element. Now, last element of the array is its max element.
        3. Reduce the size of the heap by 1. 
        4. If size of heap is more then 1 then Heapify the root again to make it a max binary heap. Else return as our 
        array has been sorted.
        5. Repeat from step number 2 till size of the heap becomes 1.
    Algorithm to Implement min/max Binary Heap:
        1. Since in a binary heap, half of the elements are at the bottom level so we will only apply heapify in top half 
        array elements.
        2. We will start with last element in 2nd last level and keep applying heapify to create a min/max binary heap.
        3. Heapify algorithm is nothing but its a recursive algorithm to apply sink method.
        4. Suppose, we called sink on node 3 of the heap/array. First, we will find its child node (2*i + 1 and 2*i + 2).
        If both child nodes are outside the size then return. Else, select the minimum of the child node to implement a 
        min binary heap. Swap this child node value with root node and apply sink on child node.    
    '''
    def create_max_binary_heap(arr):
        size = len(arr)
        half_value = (size - 1) // 2
        for r in reversed(range(half_value + 1)):
            heapify(arr, r, size - 1)

    def heapify(arr, index, max_index):
        left_child = 2*index + 1
        right_child = 2*index + 2

        if left_child > max_index: 
            return
        if right_child <= max_index and arr[right_child] > arr[left_child]:
            child = right_child
        else:
            child = left_child

        if arr[child] > arr[index]:
            temp = arr[child]
            arr[child] = arr[index]
            arr[index] = temp
            heapify(arr, child, max_index)

    if arr is None or len(arr) == 0:
        return
    
    create_max_binary_heap(arr)
    size = len(arr)
    while size > 1:
        temp = arr[0]
        arr[0] = arr[size - 1]
        arr[size - 1] = temp
        size -= 1
        heapify(arr, 0, size - 1)
    
    return arr


def bubble_sort(arr):
    '''
    Run Time of bubble_sort is O(N^2).
    It is also called as Insertion Sort.
    Algorithm:
        Run an outer loop for the whole length of the array.
        Run an inner loop also for the whole length of the array.
            Compare the current index with previous index and swap them if they are not in correct order.    
    '''
    if arr is None or len(arr) == 0:
        return

    for i in range(1, len(arr)):
        for j in range(1, len(arr)):
            if arr[j] < arr[j - 1]:
                temp = arr[j]
                arr[j] = arr[j-1] 
                arr[j-1] = temp
    return arr    
    

def selection_sort(arr):
    '''
    Selection sort is also known as child's algorithm. 
    We find the minimum of the array and put it in the beginning of the array.
    Algorithm:
        Run a loop for whole array length.
        Start an inner loop which will start from outer loop variable and goes till end of the array.
            start min as outer loop index.
            Keep comparing min with inner loop index. If inner loop index is less than min then swap inner loop index and min.
        In the end of the outer loop, whole array will be sorted.
    '''
    if arr is None or len(arr) == 0:
        return

    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            min = arr[i]
            if arr[j] < min:
                temp = min
                min = arr[j]
                arr[j] = temp
            arr[i] = min
    
    return arr


if __name__ == "__main__":
    print("####### Merge Sort #######")
    print(merge_sort([6,5,4,1,2,3,0]))

    print("\n####### Quick Sort #######")
    print(quick_sort([6,5,4,1,2,3,0]))

    print("\n####### Heap Sort #######")
    print(quick_sort([6,5,4,1,2,3,0]))

    print("\n####### Bubble/Insertion Sort #######")
    print(bubble_sort([6,5,4,1,2,3,0]))

    print("\n####### Selection Sort #######")
    print(selection_sort([6,5,4,1,2,3,0]))

  


