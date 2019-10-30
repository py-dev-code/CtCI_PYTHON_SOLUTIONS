import os
from bitstring import BitArray, BitStream
from print_tree import print_tree

# Problem: 10.1 Sorted Merge: You are given two sorted arrays, A and B, where A has a large enough buffer at the
# end to hold B. Write a method to merge B into A in sorted order.
def sorted_merge(arr1, arr2, index1, index2):
    '''
    Algorithm: 
        Sorted Merge algorithm is straight forward. It takes 2 sorted arrays and then compare the first element
        of both the arrays and based on the result, it will put the minimum element in the result array.
    Implementation:
        We will apply same algorithm but with few changes.
        We have got 2 sorted arrays here with arr1 is the bigger one to hold all the elements. To mimic a fixed 
        size array, we are using a list with buffer positions set to None.
        We have got index1 and index2 which tells us the last position of each array where element is stored.
        Since we have enough buffer in arr1, we will start the Sorted Merge in reverse order. We do not need to 
        remove elements from any array as elements of arr1 will be overwritten by the correct element during 
        iterations. We just need to take care of index movements properly.
    '''
    index_merged = index1 + index2 + 1
    while index2 >= 0:
        if index1 >= 0 and arr1[index1] > arr2[index2]:
            arr1[index_merged] = arr1[index1]
            index1 -= 1
        else:
            arr1[index_merged] = arr2[index2]
            index2 -= 1
        index_merged -= 1
    

# Problem: 10.2 Group Anagrams: Write a method to sort an array of strings so that all the anagrams are next to
# each other.
def group_anagrams(group):
    '''
    Algorithm: Its a straight forward List sort except we need to modify the sort key a bit. 
    Each string element is a String and key is a sorted version of this string.
    We can use lambda function to define the key.
    '''
    if group is None or len(group) == 0:
        return
    group.sort(key = lambda x: sorted(x))
    return group


# Problem: 10.3 Search in Rotated Array: Given a sorted array of n integers that has been rotated an unknown
# number of times, write code to find an element in the array. You may assume that the array was
# originally sorted in increasing order.
# EXAMPLE
# lnput:find 5 in {l5, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14}
# Output: 8 (the index of 5 in the array)
def search_rotated_array(arr, value):
    '''
    Algorithm:
        This search is a modified version of Binary Search. We need to handle some additional scenarios.
		Base Case:
			If arr[mid] == value, our search is completed.
		Cases when array only has unique values:
			Case#1: When min element is lower than mid element.
				Ex: [10, 15, 20, 0, 5], value to be searched is 5.
				In this case, we know that lower half of the array is normally sorted so 2nd condition is to check if input value
				is between min and mid element. If yes then we will only search in left part of the array. If no then we will 
				only search in right part of the array. We are checking value between min and mid because in a rotated array, we 
				can not be sure if value will be in the left half even if that half is normally sorted like in given example.
			Case#2: When min element is more than mid element.
				Ex: [50, 5, 20, 30, 40], value to be searched is 5.
				This is the case when right half is normally sorted. We will put the similar logic as case1.
		Cases when array can have duplicate values:
			If array only has unique values then case1 and case2 will be able to handle the complete search. Following 2 cases need to be
			added in a duplicate array.
			Case#3: When min element is same as mid element.
				In this case, we can say that either left half or right half has same values but we dont know that yet.
				Case#3A: When mid element is not same as max element.
					Ex: [2,2,2,3,4,5], value: 3
					If this is the case then we can say right half has some different values so we will search only in right half.
				Case#3B: When mid element is also same as max element.
					Ex: [2,2,2,3,4,2], value: 3
					In this case, we dont know where to search so we will search in both the halves of the array.
    '''
    def search_rotated_array_util(arr, value, min, max):
        if min > max:
            return -1
        mid = (min + max) // 2
        if arr[mid] == value:
            # Base Case
            return mid
        
        if arr[min] < arr[mid]:
            # Case1: Left half is normally sorted
            if arr[min] <= value and arr[mid] > value:
                return search_rotated_array_util(arr, value, min, mid - 1)
            else:
                return search_rotated_array_util(arr, value, mid + 1, max)
        elif arr[min] > arr[mid]:
            # Case2: Right half is normally sorted
            if arr[max] >= value and arr[mid] < value:
                return search_rotated_array_util(arr, value, mid + 1, max)
            else:
                return search_rotated_array_util(arr, value, min, mid - 1)
        elif arr[min] == arr[mid]:
            # Left half or Right half or both halves are all repeats
            if arr[mid] != arr[max]:
                # Case3A: Right half is different so search it
                return search_rotated_array_util(arr, value, mid + 1, max)
            else: 
                # Case3B: we cannot be sure where the repeat values are so we need to search both halves
                result = search_rotated_array_util(arr, value, min, mid - 1)
                if result == -1:
                    result = search_rotated_array_util(arr, value, mid + 1, max)
                return result
        
    return search_rotated_array_util(arr, value, 0, len(arr) - 1)


# Problem: 10.4 Sorted Search, No Size: You are given an array-like data structure Listy which lacks a size
# method. It does, however, have an elementAt(i) method that returns the element at index i in
# O(1) time. If i is beyond the bounds of the data structure, it returns -1. (For this reason, the data
# structure only supports positive integers.) Given a Listy which contains sorted, positive integers,
# find the index at which an element x occurs. If x occurs multiple times, you may return any index.
class Listy(object):
    def __init__(self, data):
        if not isinstance(data, list):
            raise ValueError("Invalid Input Data")
        self.d = {}
        for k, v in enumerate(data):
            self.d[k+1] = v
    def elementAt(self, index):
        try:
            result = self.d[index]
        except:
            result = -1
        return result

def sorted_search_no_size(listy, value):
    '''
    Data Structure Implementation:
        To mimic Listy Data structure, we have created a Listy class which can be initiated with a Sorted List. 
        To implement the elementAt method in O(1) time, we will store the List in a Dictionary where keys will be indexes
        and values will be actual List values. To perform the search easily, we implement the keys starting from 1.
    Algorithm:    
        We can't do linear search as it will be too straight forward and too inefficient.
        In order to perform Binary Search, we need to know the size of the array. We can determine the size of the array by using 
        elementAt method in O(logN) time. Once the size is known, we can perform a regular Binary Search.
        Few additional validations:
            1. Determining the Size: We will start with the index 1 and keep doubling it unless elementAt new index is -1 or
            less than the given value.
            2. Binary Search: Binary Search will start from index // 2 and index. We will search the lower half when 
            elementAt(mid) is -1 or elementAt(mid) is more than the value. Else, we will search in the right half.
    '''
    def binary_search(listy, value, low, high):
        if low > high:
            return 
        mid = (low + high) // 2 
        if listy.elementAt(mid) == value:
            return mid
        elif listy.elementAt(mid) == -1 or listy.elementAt(mid) > value:
            return binary_search(listy, value, low, mid - 1)
        else:
            return binary_search(listy, value, mid + 1, high)
    
    index = 1
    while listy.elementAt(index) != -1 and listy.elementAt(index) <= value:
        index *= 2
    return binary_search(listy, value, index//2, index)


# Problem: 10.5 Sparse Search: Given a sorted array of strings that is interspersed with empty strings, write a
# method to find the location of a given string.
# EXAMPLE
# Input: ball,{"at","","","", "ball", "","", "car", "","", "dad", "",""}
# Output:4
def sparse_search(array, value):
    '''
    There are 2 small modifications of Binary Search algorithm to implement the Sparse Search.
    Algorithm 1:
        If mid element is not blank then we will choose left half / right half or return the mid element as per normal Binary
        Search pattern.
        If mid element is blank then we will search the left half first and if value is not found there then we will 
        search in right half.
        This is a very small modification and will give accurate results.
    Algorithm 2:
        We will keep the binary search logic as is but to make it work, we will shift the mid to left or right point which is 
        not blank.
        If mid is not blank then mid will stay as is.
        If mid is blank, left pointer will be mid - 1 and right pointer will be mid + 1.
        Run a loop to find out new mid:
            if left < low and right > high then return -1.
            elif left >= low and left != '' then mid will be left and break the loop.
            same for right pointer.
            decrease left by 1 and increase right by 1 after the iteration.
    '''
    def sparse_search_util(array, low, high, value):
        if low > high:
            return -1
        mid = (low + high) // 2
        if array[mid] == value:
            return mid
        elif array[mid] != '' and array[mid] > value:
            return sparse_search_util(array, low, mid - 1, value)
        elif array[mid] != '' and array[mid] < value:
            return sparse_search_util(array, mid + 1, high, value)
        else:
            result = sparse_search_util(array, low, mid - 1, value)
            if result == -1:
                return sparse_search_util(array, mid + 1, high, value)
            else:
                return result
                
    if array is None or len(array) == 0 or value == '':
        return -1
    return sparse_search_util(array, 0, len(array) - 1, value)

def sparse_search2(array, value):
    def sparse_search2_util(array, low, high, value):
        if low > high:
            return -1

        mid = (low + high) // 2
        if array[mid] == '':            
            left = mid - 1
            right = mid + 1
            while True:
                if left < low and right > high:
                    return -1
                elif left >= low and array[left] != '':
                    mid = left
                    break
                elif right <= high and array[right] != '':
                    mid = right
                    break
                left -= 1
                right += 1
        
        if array[mid] == value:
            return mid
        elif array[mid] > value:
            return sparse_search2_util(array, low, mid - 1, value)
        else:
            return sparse_search2_util(array, mid + 1, high, value)

    if array is None or len(array) == 0 or value == '':
        return -1
    return sparse_search2_util(array, 0, len(array) - 1, value)


# Problem: 10.6 Sort Big File: Imagine you have a 20 GB file with one string per line. Explain how you would sort
# the file.
def external_heap_sort(input_file, memory_limit):
    '''
        Sorting a very large file by dividing it into smaller parts is known as External Sort. It can be implemented by 
        various ways.
        1. External Merge Sort:
            Divide the file into n smaller files and sort each file with any sorting algorithm.
            Merge the n sorted files into one larger output file.
        2. External Heap Sort:
            Divide the file into n smaller files.
            Sort individual files in memory by using any sorting mechanism. We will just take file data into a list, sort that list
            and write it back to the file.
            Create a min Binary heap by taking first element from each file.
            Move the root of Binary Heap into the output file.
            Go to the file from where root was taken and take its next element and add it into the heap root. Heapify the root.
            If file has already ended then put None into the Heap root and apply Heapify in a way that None will move to the end
            of the Heap.
            Keep doing this till Heap root is None. Output file will be completely sorted.
        We will implement External Heap Sort Algorithm here.        
    '''
    class HeapNode(object):
        def __init__(self, data, file_pointer):
            self.data = data
            self.file_pointer = file_pointer
        def __lt__(self, other):
            return self.data < other.data
        def __gt__(self, other):
            return self.data > other.data
        def __eq__(self, other):
            return self.data == other.data
        def __repr__(self):
            return str(self.data)

    def create_min_binary_heap(arr):
        size = len(arr)
        half_value = (size - 1) // 2
        for r in reversed(range(half_value + 1)):
            heapify(arr, r, size - 1)

    def heapify(arr, index, max_index):
        left_child = 2*index + 1
        right_child = 2*index + 2

        if left_child > max_index: 
            return
        if right_child <= max_index and arr[right_child] < arr[left_child]:
            child = right_child
        else:
            child = left_child

        if arr[child] < arr[index]:
            temp = arr[child]
            arr[child] = arr[index]
            arr[index] = temp
            heapify(arr, child, max_index)

    def sort_file(input_file):
        l = []
        with open(input_file, 'r') as f:
            l = f.readlines()
        l.sort()
        with open(input_file, 'w') as f:
            f.writelines(l)

    # Dividing the File. Keep creating the files of given size till we reach the end.
    with open(input_file, 'r') as input:        
        file_num = 1
        size = 0
        create_new_file = False
        chunk = open(str(file_num) + '.txt', 'w')

        for line in input:
            if create_new_file:
                create_new_file = False
                file_num += 1
                size = 0
                chunk.close()
                chunk = open(str(file_num) + '.txt', 'w')
            chunk.write(line)
            size += 1
            if size >= memory_limit:
                create_new_file = True
        chunk.close()

    pointer_list = []

    for r in range(1, file_num + 1):
        file_name = str(r) + '.txt'
        sort_file(file_name)
        
    for r in range(1, file_num + 1):
        pointer = open(str(r) + '.txt', 'r')
        pointer_list.append(pointer)

    heap_array = []
        
    for r in pointer_list:
        heap_array.append(HeapNode(r.readline().split('\n')[0], r))
    
    create_min_binary_heap(heap_array)
    output = open('output.txt', 'w')

    while heap_array[0].data != 'zzzz':
        # We are using split otherwise new line character was giving some issues. By using split, we are making sure that, each 
        # time we are appending new line character.
        output.write(heap_array[0].data.split('\n')[0] + '\n')
        value = heap_array[0].file_pointer.readline()
        if value == '' or value is None:
            # We can use None as well here. Then, we just need to modify lt, gt and eq methods in HeapNode class to make None
            # the maximum value.
            heap_array[0].data = 'zzzz'
        else:
            heap_array[0].data = value
        heapify(heap_array, 0, len(heap_array) - 1)
        
    for r in pointer_list:
        r.close()    

    for r in range(1, file_num + 1):
        file_name = str(r) + '.txt'
        os.remove(file_name)

    output.close()


# Problem: 10.7 Missing Int: Given an input file with four billion non-negative integers, provide an algorithm to
# generate an integer that is not contained in the file. Assume you have 1 GB of memory available for
# this task.
# FOLLOW UP
# What if you have only 10 MB of memory? Assume that all the values are distinct and we now have
# no more than one billion non-negative integers.
def get_missing_int_1GB(input_file):
    '''
    Algorithm:
        An Integer is represented as 32 bits inside the memory so we know that there can be maximum of 2**32 distinct integers
        possible which is 4294967296 (roughly 4 billion). Out of these 4 billion, half will be negative integers so we can 
        have a maximum of 2**31 distinct positive integers which will have .
        If our file contains 4 billion non-negative integers then for sure it contains some duplicate integers.
        We have 1 GB memory available which is roughly 8 billion bits.
        How we will solve the problem with 1 GB memory:
            We will create a BitArray of 2 billion bits (2**31) with all initial values set to 0. 
            We will read each line from the file and set the corresponding index in BitArray to True.
            We will loop through the BitArray and return the first index that is False.
    '''
    barray = BitArray('0b0') * 2**31
    with open('integer_file.txt', 'r') as input:
        for line in input:
            barray[int(line.split('\n')[0])] = '0b1'
    
    for index, value in enumerate(barray):
        if not value:
            return index

def get_missing_int_10MB(input_file):
    '''
    Algorithm:
        Handling the complete input file within 10MB memory leads to some interesting adjustments. We will still follow the earlier
        Bit Array approach but this time our Bit Array will not be able to represent the complete list of integeres.                
        1. We will read the complete file and create a list of integers but why?
            First, we will determine the range of numbers that each of the list integer will represent. Lets say, range of numbers
            is 1000 then after reading the complete file; we should have a list with integer values.
            If a number is missing between a range say 1001 to 1999 then 1st integer in the list will have a value less than 1000.
            This will tell us that our first missing number is from 1001 to 1999.
        2. Once the list is ready, we will find the list index whose value is less than range of numbers.
        3. Once we get the list index, we will convert those range of numbers into a Bit Array.
        4. Once Bit Array is ready, we will find the first index that is 0 and based on the list index and bit array index, we will
        calculate our missing integer.

    Analysis to determine the range of numbers and size of list:
        We can consume 10MB memory which is roughly 2**26 bits and each list element is an integer that will take 2**5 bits (32).
        This tells us that our list can contain at max 2**21 integers. Now, we can come up with range of numbers that each integer 
        can handle.
            list_size <= 2**21
            2**31 / range_of_numbers <= 2**21
            range_of_number >= 2**10
        Therefore, 1 integer needs to handle minium of 2**10 integer values.
        This is the lower limit of range_of_numbers and our upper limit is already set to 2**26 bits.
        So, we can select a value between 2**10 and 2**26 as range of numbers and we will be able to handle the complete file.

    Following function implements the same algorithm with range of numbers as 8 and list size of 10 to showcase that the 
    approach will be able to calculate the first missing number perfectly.
    '''
    def get_count_per_block(input_file, range_size):
        result = [0] * 10
        with open(input_file, 'r') as input:
            for line in input:
                value = int(line.split('\n')[0])
                index = value // range_size
                result[index] += 1
        return result

    def get_missing_value_block(block_list, range_size):
        for index, value in enumerate(block_list):
            if value < range_size:
                return index

    def get_block_bit_array(input_file, block_index, range_size):
        barray = BitArray('0b0') * range_size
        start_number = block_index * range_size
        end_number = start_number + range_size
        with open(input_file, 'r') as input:
            for line in input:
                value = int(line.split('\n')[0])
                if value >= start_number and value < end_number:
                    barray_index = value - start_number
                    barray[barray_index] = '0b1'
        return barray

    def find_zero(barray):
        for index, value in enumerate(barray):
            if not value:
                return index

    range_size = 1 << 3 # 2**3
    block_list = get_count_per_block(input_file, range_size)
    # print(block_list) # [8, 10, 3, 0, 0, 0, 0, 0, 0, 0]

    block_index = get_missing_value_block(block_list, range_size)
    # print(block_index)  # 2

    barray = get_block_bit_array(input_file, block_index, range_size)
    # print(barray.bin)   # 11001000

    offset = find_zero(barray)
    return block_index * range_size + offset


# Problem: 10.8 Find Duplicates: You have an array with all the numbers from 1 to N, where N is at most 32,000. The
# array may have duplicate entries and you do not know what N is. With only 4 kilobytes of memory
# available, how would you print all duplicate elements in the array?
def find_duplicates_4KB(array):
    '''
    Algorithm:
        Since we have been given 4KB of memory which is 4 * 8 * 1024 bits which is more than 32000. So, we can handle 
        maximum limit of given integers (which is 32000) in given memory limit.
        Implementation:
            We will create a Bit Array of 32000 size and set each bit to 0 initially.
            We will loop through the array, if index position at given number is 0 then we set the index to 1.
            If index position is already 1 that means its duplicate entry then we will just print it.
    '''
    barray = BitStream('0b0') * 32000
    for r in array:
        if barray[r - 1]:
            print(r)
        else:
            barray[r - 1] = '0b1'


# Problem: 10.9 Sorted Matrix Search: Given an M x N matrix in which each row and each column is sorted in
# ascending order, write a method to find an element.
def search_sorted_matrix(matrix, r1, c1, r2, c2, value):
    '''
    Algorithm:
        Naive Approach:
            1. We can binary search each row and get the index for given value. This will take O(NlogM) time.
            2. We can filter out some rows and columns to limit our binary search to make it more efficient.
        Optimized Approach - Binary Search on complete matrix:
            1. Start with the diagonal, we will find 2 diagonal points such that 1st point is less than value and next
            point is more than the value. Say this is (1,1) and (2,2) in a 4x4 matrix.
            2. Now based on this, we can divide the matrix in 4 sub-matrix and since matrix rows and columns are sorted
            so following properties will be applied.
                matrix1: (0,0) to (1,1) -> All elements in this matrix will be lesser than the value.
                matrix2: (2,0) to (3,1)
                matrix3: (0,2) to (1,3)
                matrix4: (2,2) to (3,3) -> All elements in this matrix will be larger than the value.
            3. Therefore, we can discard matrix1 and matrix4 and continue our search in matrix2 and matrix3. 
            4. We can call this approach recursively till we get a matrix of just 1 element.
            Base Cases:
                1. If (r1,c1) and (r2,c2) are same then compare the element and return.
                2. If (r1,c1) is more than the value then return -1 as matrix cant have the value.
                3. If (r1,c1) is None then return -1.
                4. If value matches with diagonal element then return the index and return.
            5. We just need to be careful while calculating the diagnol points in the matrix and determining the 
            dimensions of matrix2 and matrix3. Other than these, rest of the algorithm is straight forward.
    '''
    def get_element(matrix, i, j):
        try:
            return matrix[i][j]
        except:
            return None

    if r1 == r2 and c1 == c2 and get_element(matrix, r1, c1) != value:
        return -1
    if get_element(matrix, r1, c1) is not None and get_element(matrix, r1, c1) > value:
        return -1
    if get_element(matrix, r1, c1) is None:
        return -1
    
    diag_r1 = diag_r2 = r1
    diag_c1 = diag_c2 = c1
    
    while get_element(matrix, diag_r1, diag_c1) is not None:
        elem = get_element(matrix, diag_r1, diag_c1)
        if elem == value:
            return [diag_r1, diag_c1]
        elif elem < value:
            diag_r1 = diag_r2
            diag_c1 = diag_c2
            diag_r2 += 1
            diag_c2 += 1
        elif elem > value:
            break
    diag_r1 -= 1
    diag_c1 -= 1
    diag_r2 -= 1
    diag_c2 -= 1
    
    # print(diag_r1, diag_c1, diag_r2, diag_c2)
    
    left_m_r1 = diag_r2
    left_m_c1 = c1
    left_m_r2 = r2
    left_m_c2 = diag_c1

    right_m_r1 = r1
    right_m_c1 = diag_c2
    right_m_r2 = diag_r1
    right_m_c2 = c2

    # print(left_m_r1, left_m_c1, left_m_r2, left_m_c2)
    # print(right_m_r1, right_m_c1, right_m_r2, right_m_c2)
    # return

    result = search_sorted_matrix(matrix, left_m_r1, left_m_c1, left_m_r2, left_m_c2, value)
    if result == -1:
        return search_sorted_matrix(matrix, right_m_r1, right_m_c1, right_m_r2, right_m_c2, value)
    else:
        return result


# Problem: 10.10 Rank from Stream: Imagine you are reading in a stream of integers. Periodically, you wish to be able
# to look up the rank of a number x (the number of values less than or equal to x). lmplement the data
# structures and algorithms to support these operations. That is, implement the method track ( int
# x), which is called when each number is generated, and the method getRankOfNumber(int
# x), which returns the number of values less than or equal to x (not including x itself).
# EXAMPLE
# Stream (in order of appearance): 5, 1, 4, 4, 5, 9, 7, 13, 3
# getRankOfNumber(1) = 0
# getRankOfNumber(3) = 1
# getRankOfNumber(4) = 3
'''
    Algorithm:
        1. To find the rank of an element in a stream, we can sort the elements in a sorted order then do a Binary search for the given element and return the index and it will be the rank of the element. In this method, rank method will take O(logN)
        time but inserting a new element will take a lot of time as we will have to shift the elements very often.
        2. Instead of storing elements in a normal array, we can use a Binary Search Tree which will store the element in O(logN)
        time and in order to find the rank of the element, we can store an additional attribute in each node that will save 
        the number of nodes in left subtree of a given node. So, we can do a Binary search for the given element and based on
        the value of left subtree nodes, we will be able to find the rank in O(logN) time.
        Of course, to acheive the performance of O(logN) time, our Binar Search Tree should be Balanced.
        3. Below implementation is given with a normal BST as adjusting number of left nodes in a Balanced BST is quite tricky.
        4. Class gives a full implementation of Balanced BST for reference but for implementing Rank and Insert methods, balance method is not 
        being called.
'''
class Node(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        self.left_nodes = 0
    def __repr__(self):
        return str(self.data)

class BalBST(object):
    def __init__(self, root=None):
        self.root = root

    def height(self, root):
        if root is None:
            return 0
        return max(self.height(root.left), self.height(root.right)) + 1

    def balance_node(self, root):
        if root is None:
            return
        bf = self.height(root.left) - self.height(root.right)

        if bf == 2 and root.left.right is None:
            return self.rotate_right(root)
        elif bf == 2 and root.left.right is not None:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)
        elif bf == -2 and root.right.left is None:
            return self.rotate_left(root)
        elif bf == -2 and root.right.left is not None:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)
        else:
            return root

    def rotate_left(self, root):
        node = root.right
        root.right = node.left
        node.left = root
        return node

    def rotate_right(self, root):
        node = root.left
        root.left = node.right
        node.right = root
        return node
    
    def insert_key(self, key):
        self.root = self.insert_key_util(self.root, key)
    def insert_key_util(self, root, key):
        if root is None:
            return Node(key)
        if root.data >= key:
            root.left = self.insert_key_util(root.left, key)
            root.left_nodes += 1
        elif root.data < key:
            root.right = self.insert_key_util(root.right, key)
        return root
        # return self.balance_node(root)

    def get_rank(self, value):
        return self.get_rank_util(self.root, value)
    def get_rank_util(self, root, value):
        if root.data == value:
            return root.left_nodes
        elif root.data > value:
            if root.left is None: 
                return -1
            else: 
                return self.get_rank_util(root.left, value)
        elif root.data < value:
            if root.right is None:
                right_rank = -1
            else:
                right_rank = self.get_rank_util(root.right, value)
            if right_rank == -1:
                return -1
            else:
                return root.left_nodes + 1 + right_rank

    def remove_key(self, key):
        self.root = self.remove_key_util(self.root, key)
    def remove_key_util(self, root, key):
        if root is None:
            return None
        if root.data > key:
            root.left = self.remove_key_util(root.left, key)
        elif root.data < key:
            root.right = self.remove_key_util(root.right, key)
        else:
            # case 1,2,3: When left node or right node or both the nodes are None.
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            # case 4: when both nodes are not None
            else:
                node = root.right
                while node.left:
                    node = node.left
                temp = root.data
                root.data = node.data
                node.data = temp
                root.right = self.remove_key_util(root.right, key)
        return self.balance_node(root)


# Problem: 10.11 Peaks and Valleys: In an array of integers, a "peak" is an element which is greater than or equal to
# the adjacent integers and a "valley" is an element which is less than or equal to the adjacent integers.
# For example, in the array {5, 8, 6, 2, 3, 4, 6}, {8, 6} are peaks and {5, 2} are valleys. Given an array
# of integers, sort the array into an alternating sequence of peaks and valleys.
# EXAMPLE
# Input: {5, 3, 1, 2, 3}
# Output: {5, 1, 3, 2, 3}
def sort_valley_and_peak(arr):
    '''
    Algorithm:
        1. We will loop through the array starting from index 1 and skipping 1 element each time that is 1,3,5,7 and so on.
        2. We will pick both the adjacent elements for the loop index and adjust those 3 values. If last element is out of 
        index then we will just adjust the 2 elements.
        3. In order to put a valley in between, we will create a sorted list of the 3 elements and assign r index to the 1st
        element.
        4. r-1 will be the next element and r+1 will be the last element from the sorted list.
        5. Once done, out input array will be sorted as per the peaks and valleys.
        6. In order to sort the array with valleys and peaks, we can assign max element from sorted list to r, min element to r-1
        and remaining element to r+1.
        7. This is an optimal approach and will take O(N) time.
    '''
    
    if arr is None or len(arr) == 0:
        return
    
    for r in range(1,len(arr),2):
        if r == len(arr) - 1:
            temp = sorted([arr[r-1], arr[r]])
            arr[r] = temp.pop(0)
            arr[r-1] = temp.pop()
            
        else:
            temp = sorted([arr[r-1], arr[r], arr[r+1]])
            arr[r] = temp.pop(0)
            arr[r-1] = temp.pop()            
            arr[r+1] = temp.pop()
    
    return arr


if __name__ == "__main__":
    print("Problem# 10.1")
    arr1 = [0,2,4,None,None]
    arr2 = [1,3]
    sorted_merge(arr1, arr2, 2, 1)
    print(arr1)

    print("\nProblem# 10.2")
    g = ['abc', 'cba', 'bac', 'acd', 'def']
    print(group_anagrams(g))

    print("\nProblem# 10.3")
    arr1 = [10, 15, 20, 0, 5]   # Case1
    arr2 = [50, 5, 20, 30, 40]  # Case2
    arr3 = [2,2,2,3,4,5]        # Case3A
    arr4 = [2,2,2,3,4,2]        # Case3B
    print(search_rotated_array(arr1, 5))
    print(search_rotated_array(arr2, 5))
    print(search_rotated_array(arr3, 3))
    print(search_rotated_array(arr4, 3))

    print("\nProblem# 10.4")
    l = [1,2,3,4,5,6,7,8,8,9,10]
    listy = Listy(l)
    for r in range(12):
        print(f'{r}: {sorted_search_no_size(listy, r)}')

    print("\nProblem# 10.5")
    arr = ['at', '', '', '', 'ball', '', '', 'car', '', '', 'dad', '', '']
    print(sparse_search(arr, 'dad'))
    print(sparse_search2(arr, 'dad'))

    print("\nProblem# 10.6")
    # Change the directory to current directory before executing the following function.
    # external_heap_sort('input.txt', 4)

    print("\nProblem# 10.7")
    # Change the directory to current directory before executing the following function.
    # print(get_missing_int_1GB('integer_file.txt'))
    # print(get_missing_int_10MB('integer_file.txt'))

    print("\nProblem# 10.8")
    find_duplicates_4KB([1,2,3,1,2,3,4,5,4,1])

    print("\nProblem# 10.9")
    matrix = [
        [1, 2,  3,  4,  4.5],
        [5, 6,  7,  8,  8.5],
        [9, 10, 11, 12, 12.5],
        [13,14, 15, 16, 16.5]
    ]
    print(search_sorted_matrix(matrix, 0, 0, 3, 4, 5))

    print("\nProblem# 10.10")
    bst = BalBST()
    for r in [4,3,1,2,7,6,5,8,9]:
        bst.insert_key(r)
    print_tree(bst.root, bst.height(bst.root), 2)
    print(bst.get_rank(9))

    print("\nProblem# 10.11")
    print(sort_valley_and_peak([5,3,1,2,3]))