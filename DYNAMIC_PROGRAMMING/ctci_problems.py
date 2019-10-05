from collections import Counter

# Problem:8.1 Triple Step: A child is running up a staircase with n steps and can hop either 1 step, 2 steps, or 3
# steps at a time. Implement a method to count how many possible ways the child can run up the
# stairs.
def count_triple_step_ways(n):
    # Used Approach: Bottom Up
    if n < 0:
        return 0
    l = [None for x in range(max(3, n+1))]
    # Ways to take 0 steps can be taken either 0 or 1 which will give different outputs.
    l[0] = 0    
    l[1] = 1
    l[2] = 2
    for r in range(3, n+1):
        l[r] = l[r-1] + l[r-2] + l[r-3]
    return l[n]


# Problem:8.2 Robot in a Grid: Imagine a robot sitting on the upper left corner of grid with r rows and c columns.
# The robot can only move in two directions, right and down, but certain cells are "off limits" such that
# the robot cannot step on them. Design an algorithm to find a path for the robot from the top left to
# the bottom right.
def get_grid_path(matrix):

    def get_grid_path_util(matrix, r, c, path):
        point = (r,c)
        path.append(point)
        if r == row - 1 and c == col - 1:
            return True
        # Down Cell
        x_down = r+1
        y_down = c
        if x_down < row and y_down < col and matrix[x_down][y_down] == 1:
            return get_grid_path_util(matrix, x_down, y_down, path)    
        # Right Cell
        x_right = r
        y_right = c+1
        if x_right < row and y_right < col and matrix[x_right][y_right] == 1:
            return get_grid_path_util(matrix, x_right, y_right, path)        
        return False

    if matrix is None or len(matrix) == 0:
        return []
    row = len(matrix)
    col = len(matrix[0])
    path = []
    if get_grid_path_util(matrix, 0, 0, path):
        return path
    else:
        return "No path possible."
 

# Probelm:8.3 Magic Index: A magic index in an array A[ 1 .•. n-1] is defined to be an index such that A[ i] = i. 
# Given a sorted array of distinct integers, write a method to find a magic index, if one exists, in
# array A.
# FOLLOW UP
# What if the values are not distinct?
def get_magic_index(array):
    '''
    Function to find magic index when input array doesnt have duplicate values.
    '''    
    def get_magic_index_util(array, low, high):
        if low > high:
            return None
        mid = (low + high) // 2
        if array[mid] == mid:
            return mid
        elif array[mid] > mid:
            return get_magic_index_util(array, low, mid - 1)
        elif array[mid] < mid:
            return get_magic_index_util(array, mid + 1, high)    
    
    if array is None or len(array) == 0:
        return None
    return get_magic_index_util(array, 0, len(array) - 1)

def get_magic_index_in_dupl_array(array):
    '''
    This function will find the magic index when given sorted array can contain duplicate values.
    We will skip few elements in recursion by following logic:
    if array[mid] == mid then return mid.
    left recursion will run from low to min(mid-1, arr[mid]). If mid is 3 and arr[mid] is 0 then we dont need to 
    check 1 and 2.
    right recursion will run from max(mid+1, arr[mid]) to high. If mid is 3 and arr[mid] is 6 then we dont need to 
    check 4 and 5.
    We will call both the recursions one after the other as index can be anywhere.
    '''
    def get_magic_index_in_dupl_array_util(array, low, high):
        if low > high:
            return None
        mid = (low + high) // 2
        if array[mid] == mid:
            return mid
        left =  get_magic_index_in_dupl_array_util(array, low, min(array[mid], mid-1))
        if left is None:
            return get_magic_index_in_dupl_array_util(array, max(array[mid], mid+1), high)
        else:
            return left

    if array is None or len(array) == 0:
        return None
    return get_magic_index_in_dupl_array_util(array, 0, len(array) - 1)


# Problem:8.4 Power Set: Write a method to return all subsets of a set.
def get_power_set(array):
    '''
    Algorithm:
    For an empty array, result will be an empty set.
    For a single element array, result will have an empty set and a set of the element.
    For subsequent lenghts, we just need to keep the next element in the list of previous level.
    We will build the result iteratively.
    '''
    if array is None or len(array) == 0:
        return set()
    result = [[] for x in range(len(array))]
    result[0].append(set())
    result[0].append({array[0]})
    for r in range(1, len(array)):
        result[r].extend(result[r-1])
        for i in result[r-1]:
            temp = set(i)
            temp.add(array[r])
            result[r].append(temp)
    return(result[len(result) - 1])

# Problem:8.5 Recursive Multiply: Write a recursive function to multiply two positive integers without using the
# * operator. You can use addition, subtraction, and bit shifting, but you should minimize the number
# of those operations.    
def multiply_without_into(x, y):
    if y == 0:
        return 0
    if y < 0:
        return - multiply_without_into(x, -y)
    result = 0
    for _ in range(y):
        result += x
    return result

def recursive_multiply(x, y):
    '''
    Algorithm:
    Rather than doing normal method (adding y for x times), we can implement a logarithmic complexity approach.
    First, we will identify smaller and bigger number so that number of recursions will be less.
    Next, If smaller number is 0 or 1, function will return the result.
    We will divide the smaller number by 2 (using bit shift operator) and call recursion.
    Trick here is to check if smaller number is even or not.
    If even, return half_sum + half_sum else return half_sum + half_sum + bigger.
    No need for memoization as we will not call the recursion for same values again.
    '''
    def recursive_multiply_util(smaller, bigger):
        if smaller < 0:
            return - recursive_multiply_util(-smaller, bigger) 
        if smaller == 0:
            return 0
        if smaller == 1:
            return bigger
        half = smaller >> 1     # This is equivalent to smaller // 2
        half_result = recursive_multiply_util(half, bigger)
        if smaller % 2 == 0:
            return half_result + half_result
        else:
            return half_result + half_result + bigger
    
    if x < y:
        smaller, bigger = x, y
    else:
        smaller, bigger = y, x
    return recursive_multiply_util(smaller, bigger)
    

# Problem:8.6 Towers of Hanoi: In the classic problem of the Towers of Hanoi, you have 3 towers and N disks of
# different sizes which can slide onto any tower. The puzzle starts with disks sorted in ascending order
# of size from top to bottom (i.e., each disk sits on top of an even larger one). You have the following
# constraints:
# (1) Only one disk can be moved at a time.
# (2) A disk is slid off the top of one tower onto another tower.
# (3) A disk cannot be placed on top of a smaller disk.
# Write a program to move the disks from the first tower to the last using stacks.
def tower_of_hanoi(n, origin, destination, buffer):
    '''
    Algorithm: Stack can be implemented by list assuming that only append and pop can be used.
    Input n tells that how many disks from origin stack needs to be moved to destination.
    Base Case: If n is 1, we will move the disk directly to destination.
    Higher Case: 
        Move n-1 disks from origin to buffer (using destination as buffer stack).
        Move 1 disk from origin to destination.
        Move n-1 disks from buffer to destination (using origin as buffer stack).
    '''
    if n <= 0:
        return
    if n == 1:
        destination.append(origin.pop())
        return
    tower_of_hanoi(n-1, origin, buffer, destination)
    tower_of_hanoi(1, origin, destination, buffer)
    tower_of_hanoi(n-1, buffer, destination, origin)
    

# Problem:8.7 Permutations without Dups: Write a method to compute all permutations of a string of unique
# characters.
def string_permutations_no_dupl(string):
    '''
    Algorithm:
    Assumption is that String does not have any duplicate characters.
    Base Case: If string length is 1 then 1st character of string will be returned.
    Higher Case (length 2): We need to add 0th character of the String to all possible place in result of base case.
    If string is 'ab' then base result will be ['b'].
    Now, we will append 'a' on 0th and 1st position of 'b' to make the final result.
    Refer the next method which is easier to implement and much faster for duplicate strings as well.
    '''
    if string is None or len(string) == 0:
        return
    n = len(string)
    if n == 1:
        return [string[0]]    
    base_result = string_permutations_no_dupl(string[1: n])
    result = []
    for r in range(n):
        for i in base_result:            
            temp = i[0: len(i) - r] + string[0] + i[len(i) - r: len(i)]
            result.append(temp)
    # To clear the base list as it is will not be used now.
    base_result.clear()
    return result
    

# Problem:8.8 Permutations with Dups: Write a method to compute all permutations of a string whose characters
# are not necessarily unique. The list of permutations should not have duplicates.
def string_permutations_by_counter(string):
    '''
    Algortihm: 1st algorithm will be similar to previous function except we will only add a string in result when it has not 
    already been added.
    Below is the very effective algorithm which will run substantially faster when strgin has a lot of duplicates.
    Implementation:
    1. Convert the string into a counter and put the recursive logic on this counter.
    2. Remove elements from Counter that has 0 or negative counts to avoid infinite loop.
    3. Base cases: If length of counter is 1 then result will be a list with 1 string with all its count. Ex:
        Counter({'a':2}) will return ['aa']
    4. Create a copy of input counter and iterate through its keys.
    5. For each key, subtract the count by 1 in input counter and call the recursive method.
    6. Append the key in returned list.
    7. Once done, increase the count of that key by 1 to keep the same counter for other iterations.
    '''    
    def get_counter_permutations(c):
        c += Counter()
        if len(c) == 0: return
        if len(c) == 1:
            for k in c.keys():
                return [k * c[k]]

        result = []
        c1 = Counter(c)

        for k in c1.keys():
            c[k] -= 1
            base_case = get_counter_permutations(c)
            for r in base_case:
                result.append(k + r)
            c[k] += 1
        return result

    if string is None or len(string) == 0:
        return
    c = Counter(string)
    return get_counter_permutations(c)


# Probelm:8.9 Parens: Implement an algorithm to print all valid (e.g., properly opened and closed) combinations
# of n pairs of parentheses.
# EXAMPLE
# Input: 3
# Output: ( ( () ) ) , ( () () ) , ( () ) () , () ( () ) , () () ()
def get_bracket_pairs(n):
    '''
    Approach 1: 
        Base Case: If n is 1 then return a Single Pair of brackets.
        Higher Case: Add a pair of brackets in all possible places and add it in the result if its not a duplicate.
        This will work but it will take a lot of duplicate checks.
        This has not been written here.
    Approach 2:
        We will create the result from scratch and keep discarding cases of invalid bracket pairs.
        Our util method will check for current string validity. If remaining left is less than 0 or remaining right is
        less than remaining left then string is invalid.
        If both left remaining and right remaining are 0 then keep string in result.
        Invoke the util method by placing '(' and ')' on input index so that all combinations will be tried out (except the
        ones where string already became invalid by placing more right brackets first).
        Since we will try each combination only once, no need to handle duplicate combinations.
        This is more efficient approach and implemented below.
    '''
    def get_bracket_pairs_util(left_rem, right_rem, index, string, result):
        if left_rem < 0 or right_rem < left_rem:
            # invalid string
            return
        if left_rem == 0 and right_rem == 0:
            result.append(''.join(pair))
            return
        pair[index] = '('
        get_bracket_pairs_util(left_rem - 1, right_rem, index + 1, pair, result)
        pair[index] = ')'
        get_bracket_pairs_util(left_rem, right_rem - 1, index + 1, pair, result)
    
    if n <= 0:
        return
    result = []
    pair = [None] * n * 2
    get_bracket_pairs_util(n, n, 0, pair, result)
    return result


# Probelm:8.10 Paint Fill: Implement the "paint fill" function that one might see on many image editing programs.
# That is, given a screen (represented by a two-dimensional array of colors), a point, and a new color,
# fill in the surrounding area until the color changes from the original color.
# input:  1   0   2   2   0
#         0   1   0   0   3
#         1   0   0   2   1
#         1   0   0   0   2
#         3   2   0   0   1
# pointer: row: 2, col: 2 [starting from 0]
# new color: 4
# output: 1   0   2   2   0
#         0   1   4   4   3
#         1   4   4   2   1
#         1   4   4   4   2
#         3   2   4   4   1

def paint_fill(screen_matrix, r, c, new_color):
    '''
    Algorithm:
        This is a straight forward matrix traversal case (or Graph traversal for that matter).
        Take the value of given cell and pass it in util method.
        If current cell value is not equal to old value then return else change it to new value.
        Call util method to left, right, up and down cells. 
    '''
    def paint_fill_util(screen_matrix, r, c, old_color, new_color):
        if row <= r or col <= c or r < 0 or c < 0:
            return
        if screen_matrix[r][c] != old_color:
            return
        screen_matrix[r][c] = new_color
        paint_fill_util(screen_matrix, r, c-1, old_color, new_color)    # left
        paint_fill_util(screen_matrix, r, c+1, old_color, new_color)    # right
        paint_fill_util(screen_matrix, r-1, c, old_color, new_color)    # up
        paint_fill_util(screen_matrix, r+1, c, old_color, new_color)    # down
    
    if screen_matrix is None or len(screen_matrix) == 0 or r < 0 or c < 0:
        return
    row = len(screen_matrix)
    col = len(screen_matrix[0])
    old_color = screen_matrix[r][c]
    paint_fill_util(screen_matrix, r, c, old_color, new_color)


# Problem:8.11 Coins: Given an infinite number of quarters (25 cents), dimes (10 cents), nickels (5 cents), and
# pennies (1 cent), write code to calculate the number of ways of representing n cents.
def make_change(cents):
    num_quarters = cents // 25
    num_dimes = cents // 10
    num_nickels = cents // 5
    num_pennies = cents

    run1 = run2 = run3 = run4 = result = 0
    
    for q in range(0, num_quarters + 1):
        if q*25 == cents:
            run1 += 1
            result += 1
            print(f'{q} Quarters')
            break

        for d in range(0, num_dimes + 1):
            if q*25 + d*10 == cents:
                run2 += 1
                result += 1
                print(f'{q} Quarters, {d} Dimes')
                break

            for n in range(0, num_nickels + 1):
                if q*25 + d*10 + n*5 == cents:
                    run3 += 1
                    result += 1
                    print(f'{q} Quarters, {d} Dimes, {n} Nickels')
                    break
    

                for p in range(0, num_pennies + 1):
                    if p*1 + n*5 + d*10 + q*25 == cents:
                        run4 += 1
                        result += 1
                        print(f'{q} Quarters, {d} Dimes, {n} Nickels, {p} Pennies')
                        break

    print(f'Individual Executions: {run1}, {run2}, {run3}, {run4}')
    print(f'Total Executions: {run1 + run2 + run3 + run4}')
    print(f'Total possible ways: {result}')


# Problem:8.12 Eight Queens: Write an algorithm to print all ways of arranging eight queens on an 8x8 chess board
# so that none of them share the same row, column, or diagonal. In this case, "diagonal" means all
# diagonals, not just the two that bisect the board.
def place_queens(board_size):
    '''
    Algorithm:
    This one is quite complicated. First of all, we will not maintain a matrix as each row will only have 1 
    not None value so result can be represented by a simple list.
    For validating the position to place a queen, we will validate only column and diagonal as input row
    will always be empty.
    Rest is to hanlde the recursion call and we can have our results.
    '''    
    def place_queens_util(row, col_list, results):   
        
        def is_valid(row, col, col_list):
            for r in range(row):
                c = col_list[r]
                if col == c:
                    return False
                col_dist = abs(c - col)
                row_dist = abs(r - row)
                if col_dist == row_dist:
                    return False
            return True            
            
        if row == board_size:
            results.append(list(col_list))
        else:
            for col in range(board_size):                
                if is_valid(row, col, col_list):
                    col_list[row] = col
                    place_queens_util(row + 1, col_list, results)
    
    col_list = [None] * board_size
    results = []
    place_queens_util(0, col_list, results)
    for r in results:
        print(r)


# Problem:8.13 Stack of Boxes: You have a stack of n boxes, with widths wi , heights hi, and depths di. The boxes
# cannot be rotated and can only be stacked on top of one another if each box in the stack is strictly
# larger than the box above it in width, height, and depth. Implement a method to compute the
# height of the tallest possible stack. The height of a stack is the sum of the heights of each box.
def stack_of_boxes(tuple_list):
    '''
    Algorithm:
    We will create a class Box to handle all the dimensions. Function can take a list of Tuples that will be used
    to create a list of Boxes. We will sort this List (on the basis of any dimension) to make the calculation easier.
    It will work becuase if Box 1 is higher in any dimension than Box 2 then it can never go on top of Box 2.
    Algorithm will keep every Box in bottm and try check the height with maximum found height.
    Util method will be called recursively. Important part is to put a break in for loop of util function as recursion
    will calculate the complete height for a given Box number at bottom.
    We are also using a Dictionary to save the height for an individual Box number.    
    In addition to calculate the maximum height, function will also print the Boxes that will be used to create the 
    Stack.
    '''
    class Box(object):
        def __init__(self, dim_tuple):
            self.w = dim_tuple[0]
            self.h = dim_tuple[1]
            self.l = dim_tuple[2]
        def __repr__(self):
            return f'{self.w}W:{self.h}H:{self.l}L'

    def can_go_on_top(bottom, top, box_list):
        return (box_list[bottom].l > box_list[top].l and
                box_list[bottom].w > box_list[top].w and
                box_list[bottom].h > box_list[top].h)

    def get_stack_height(box_num, box_list, height_map, result):
        
        if height_map[str(box_num)] is not None:
            return height_map[str(box_num)]        
        height = box_list[box_num].h
        result.append(box_num)
        for i in range(box_num + 1, len(box_list)):
            if can_go_on_top(box_num, i, box_list):
                height += get_stack_height(i, box_list, height_map, result)        
                break
        height_map[str(box_num)] = height
        return height
        
    box_list = []
    for r in tuple_list:
        box_list.append(Box(r))
    box_list.sort(key = lambda x: x.h, reverse = True)
    
    max_height = 0
    height_map = {}
    for r in range(len(box_list)):
        height_map[str(r)] = None

    result = []
    results = [None]        
    for r in range(len(box_list)):
        height = get_stack_height(r, box_list, height_map, result)
        if height > max_height:
            max_height = height
            results[0] = result
            result = []

    for r in results[0]:
        print(f'Box#{r}: {box_list[r]}')    
    return max_height


# Problem:8.14 Boolean Evaluation: Given a boolean expression consisting of the symbols 0 (false), 1 (true), &
# (AND), I (OR), and ^ (XOR), and a desired boolean result value result, implement a function to
# count the number of ways of parenthesizing the expression such that it evaluates to result.
# EXAMPLE
# countEval("1^0|0|1", false) -> 2
# countEval("0&0&0&1^l|0", true) -> 10
def count_boolean_eval(exp, result, cache_dict={}):
    '''
    Algorithm:
    Base Case: If length of string is 1 and result is True then 1 possible way, string is 1 and result is False then 0 ways.
    Higher Case: Split the string on each possible place and count the number of ways for evaluation.
        Ex: 1&1&1 will have 2 places to try expressions in bracket [position 1 and position 3]
    We will evaluate the ways for each position and count them to get the total possible ways.
    How to evaluate the operation ways:
        We will end up like following: left operator right, result
    We need to come up with 3 results: total_ways, total_true and total_false (total_ways - total_true)
        total_ways = (left_true + left_false) * (right_true + right_false)
        total_true (if operator is &): left_true * right_true
        total_true (if operator is ^): (left_true * right_false) + (left_right * right_false)
        similarly for |.
    In the end, we can implement memoization by saving the result of (exp, result) combination in a dictionary.    
    '''    
    if exp is None or ' ' in exp or len(exp) == 0:
        return 0    
    n = len(exp)
    if n == 1 and exp == '1' and result: 
        return 1
    elif n == 1 and exp == '0' and not result: 
        return 1
    elif n == 1: 
        return 0
    
    res_string = 'True' if result else 'False'

    try:
        val = cache_dict[exp + res_string]
        return val
    except:
        None

    ways = 0    
    sub_ways = 0

    for r in range(1,n,2):
            
        left = exp[0:r]
        right = exp[r+1:n]

        left_true = count_boolean_eval(left, True, cache_dict)
        left_false = count_boolean_eval(left, False, cache_dict)
        right_true = count_boolean_eval(right, True, cache_dict)
        right_false = count_boolean_eval(right, False, cache_dict)

        total_ways = (left_true + left_false) * (right_true + right_false)
            
        if exp[r] == '&':
            total_true = left_true * right_true
        elif exp[r] == '|':
            total_true = (left_true * right_true) + (left_false * right_true) + (left_true * right_false)
        elif exp[r] == '^':
            total_true = (left_true * right_false) + (left_false * right_true)

        if result:
            sub_ways = total_true
        else:
            sub_ways = total_ways - total_true            
         
        ways += sub_ways    

    cache_dict[exp + res_string] = ways  
    return ways
    

if __name__ == "__main__":
    # 8.1
    print("Problem 8.1")
    print(count_triple_step_ways(15))       # 4841

    # 8.2
    print("\nProblem 8.2")    
    matrix = [
        [1,1,1], 
        [1,1,1], 
        [0,0,1]]
    print(get_grid_path(matrix))

    # 8.3
    print("\nProblem 8.3")    
    print(get_magic_index([-1, 1, 3, 4, 5, 6, 8, 10]))
    print(get_magic_index_in_dupl_array([6]*8))

    # 8.4
    print("\nProblem 8.4")    
    print(get_power_set([1,2,3,4]))

    # 8.5
    print("\nProblem 8.5")
    print(multiply_without_into(-2,5))
    print(recursive_multiply(5,-6))
    
    # 8.6
    print("\nProblem 8.6")
    origin = [1,2,3,4,5,6]
    buffer = []
    destination = []
    tower_of_hanoi(6, origin, destination, buffer)
    print(f'{origin} : {buffer} : {destination}')
    
    # 8.7
    print("\nProblem 8.7")
    perm = string_permutations_no_dupl('abcd')  # Length will be 4!
    print(f'{len(perm)}: {perm}')

    # 8.8
    print("\nProblem 8.8")
    print(string_permutations_by_counter('aabc'))
    
    # 8.9
    print("\nProblem 8.9")
    print(get_bracket_pairs(3))

    # 8.10
    print("\nProblem 8.10")
    m = [
        [1, 0,  2,  2,  0],
        [0, 1,  0,  0,  3],
        [1, 0,  0,  2,  1],
        [1, 0,  0,  0,  2],
        [3, 2,  0,  0,  1]
    ]
    paint_fill(m, 2, 2, 5)
    for r in m:
        print(r)

    # 8.11
    print("\nProblem 8.11")
    make_change(25)

    # 8.12
    print("\nProblem 8.12")
    place_queens(6)    

    # 8.13
    print("\nProblem 8.13")
    print(stack_of_boxes([(6, 4, 4), (8, 6, 2), (5, 33, 3), (2, 2, 2), (1, 1, 1), (9, 7, 3)]))    
    print(stack_of_boxes([(6, 4, 4), (8, 6, 2), (5, 3, 3), (7, 8, 3), (4, 2, 2), (9, 7, 3)]))  

    # 8.14
    print("\nProblem 8.14")
    print(count_boolean_eval('0&0&0&1^1|0', True))
    print(count_boolean_eval('1&1&1', True))
    print(count_boolean_eval('1^0|0|1', False))

        


