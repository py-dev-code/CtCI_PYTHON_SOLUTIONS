# Q1: Find the total number of ways by which we can take n stairs in a step size of 1, 2 or 3
def count_stair_ways(n):
    if n < 0: return 0
    if n == 0: return 1
    if n == 1: return 1
    return count_stair_ways(n - 1) + count_stair_ways(n - 2) + count_stair_ways(n - 3)
# print(count_stair_ways(20))
def count_ways_dp1(n):  # Bottom Up Approach
    l = [0] * (n + 1)
    l[0] = 1
    l[1] = 1
    l[2] = 2
    for i in range(3, n + 1):
        l[i] = l[i - 1] + l[i - 2] + l[i - 3]
    return l[n]
# print(count_ways_dp1(35))
def count_ways_dp2(n): # Top Down approach
    l = [0] * (n + 1)
    l[0] = 1
    l[1] = 1
    l[2] = 2 
    return count_ways_memo(n, l)
def count_ways_memo(n, l):
    if l[n] != 0: return l[n]
    l[n] = count_ways_memo(n - 1, l) + count_ways_memo(n - 2, l) + count_ways_memo(n - 3, l)
    return l[n]
# print(count_ways_dp2(35))
 
# Q2: Find the path in rxc matrix. Can go only in right/bottom directions. Certain cells are out of reach.
def find_path_matrix(matrix):
    if matrix is None or len(matrix) == 0:
        return None
    path = []
    result = is_path(matrix, len(matrix) - 1, len(matrix[0]) - 1, path)
    if result: 
        return path
    else: 
        return "No Path Possible"
def is_path(matrix, r, c, path):
    if r < 0 or c < 0 or matrix[r][c] == 0:
        return False
    if r == 0 and c == 0:
        is_start_point = True
    else:
        is_start_point = False
    if is_start_point or is_path(matrix, r - 1, c, path ) or is_path(matrix, r, c - 1, path):
        point = (r,c)
        path.append(point)
        return True
    else:
        return False
# matrix = [[1,1,1], [1,1,1], [0,1,1]]
# print(find_path_matrix(matrix))

def get_path_memoized(matrix):
    if matrix is None or len(matrix) == 0: return None
    path = []
    failed_points = []
    if is_path_memo(matrix, len(matrix) - 1, len(matrix[0]) - 1, path, failed_points):
        return path
    else:
        return False
def is_path_memo(matrix, r, c, path, failed_points):
    if r < 0 or c < 0 or matrix[r][c] == 0:
        return False
    point = (r,c)
    if point in failed_points:
        return False
    if r == 0 and c == 0: 
        is_origin = True
    else: 
        is_origin = False
    if is_origin or is_path_memo(matrix, r - 1, c, path, failed_points) or is_path_memo(matrix, r, c - 1, path, failed_points):
        path.append(point)
        return True
    failed_points.append(point)
    return False
# matrix = [[1,1,1], [1,1,1], [0,1,1]]
# print(get_path_memoized(matrix))        

# Q3: Magic Index: Where value at given index is equal to the index. Array is a sorted array of unique integers. Write a method to find 
# out its magic index if one exists.
def get_magic_index(arr):
    if arr is None or len(arr) == 0:
        return False
    return get_magic_index_arr(arr, 0, len(arr) - 1)
def get_magic_index_arr(arr, low, high):
    if low > high: return False
    mid = (low + high) // 2
    if arr[mid] == mid:
        return mid
    elif arr[mid] > mid:
        return get_magic_index_arr(arr, low, mid - 1)
    elif arr[mid] < mid:
        return get_magic_index_arr(arr, mid + 1, high)
# arr = [-1, -2, 0, 1, 2, 3, 8, 10]
# print(get_magic_index(arr))

# In case, elements can be non unique then we will have to check the whole array but we can still avoid some elements by comparing the 
# element index and element value.
def get_magic_index_non_unique(arr):
    if arr is None or len(arr) == 0:
        return None
    print_magic_index_nq(arr, 0, len(arr) - 1)
def print_magic_index_nq(arr, low, high):    
    if low > high:
        return
    mid = (low + high) // 2    
    if arr[mid] == mid:
        print(mid)
    print_magic_index_nq(arr, low, min(mid - 1, arr[mid]))
    print_magic_index_nq(arr, max(mid + 1, arr[mid]), high)    
# arr = [-1, 2, 2, 3, 3, 3, 8, 10]
# get_magic_index_non_unique(arr)

# Q4: Print all the subsets of a given array.
# Printing all the subsets at each level.
def print_array_subsets(arr):
    if arr is None: return None
    n = len(arr)
    if n == 0: return arr
    l1 = [None for x in range(0, n)]
    l2 = [[] for x in l1]
    empty_set = set()
    l2[0].append(empty_set)
    l2[0].append({arr[0]})
    for i in range(1, n):
        l2[i].extend(l2[i - 1])
        for j in l2[i - 1]:
            temp = set(j)
            temp.add(arr[i])
            l2[i].append(temp)
    for i in l2:
        print(i)
# printing only the subsets of given array        
def print_array_subsets2(arr):
    if arr is None: return None
    n = len(arr)
    if n == 0: return arr
    l = []    
    empty_set = set()
    l.append(empty_set)
    l.append({arr[0]})
    for i in range(1, n):
        temp = [x for x in l]
        for j in temp:
            temp_set = set(j)
            temp_set.add(arr[i])
            l.append(temp_set)
    print(l)
# print_array_subsets([1,2,3])
# print()
# print_array_subsets2([1,2,3])


        


