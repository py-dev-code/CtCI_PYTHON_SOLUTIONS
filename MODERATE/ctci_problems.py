from bitstring import BitArray
from collections import namedtuple, deque, Counter
import math
import random

# Problem 16.1: Number Swapper: Write a function to swap a number in place (that is, 
# without temporary variables).
'''
    Algorithm:
        1. First is to use (a+b) to store in a/b.
        2. Second is to use XOR operator.
'''
def number_swapper(a, b):
    # a = a + b
    # b = a - b
    # a = a - b
    a = a ^ b
    b = a ^ b
    a = a ^ b
    print(a, b)

if __name__ == "__main1__":
    number_swapper(1, 2)


# Problem 16.2: Word Frequencies: Design a method to find the frequency of occurrences of 
# any given word in a book. What if we were running this algorithm multiple times?
'''
    Algorithm:
        1. We will have to scan the entire book to get the initial frequencies. Trim and lower case the words and ignore
        all the characters other than alphabets.
        2. To support the repetitive queries, we can store the results in a Hash Table. 
'''
class Book(object):
    def __init__(self, book):
        if book is None:
            raise ValueError("Book cannot be Empty.")
        self.hash = self.setup_hash(book)
    
    def setup_hash(self, book):
        hash = {}
        for word in [x for x in book.split(' ') if x != '']:
            word = word.lower().strip()
            word = ''.join([x for x in word if ord(x) >= ord('a') and ord(x) <= ord('z')])
            if word in hash:
                hash[word] += 1
            else:
                hash[word] = 1            
        return hash

    def get_frequency(self, word):        
        if word.lower() in self.hash:
            return self.hash[word]
        else:
            return 0
        
if __name__ == "__main1__":
    book = Book("I am a   disco dancer.! dancer")
    print(book.get_frequency('dancer'))

    
# Problem 16.3: Intersection: Given two straight line segments (represented as a start point and an end point),
# compute the point of intersection, if any.
'''
    Algorithm:
        1. Inputs: 4 points, P1 P2 P3 P4. Each is a tuple with x and y coordinates.
        2. Calculate the Line Slope and y Intercept for both the lines.
        3. To make the calculations easier, we will rearrange P1 P2 P3 P4 in increasing order of their x coordinates.
        3. If lines have same slope, return an Exception. In this case, if both the lines are same line then we can 
        even return P3 if it lies in between P1 and P2. We are doing 2nd approach.
        4. Return the intersection point if it falls between the given segments.
'''
Point = namedtuple('Point', ['x', 'y'])

def get_intersection_point(p1, p2, p3, p4):
    # Arranging the points in sequence to make calculations easier
    if p1.x > p2.x: p1, p2 = p2, p1
    if p3.x > p4.x: p3, p4 = p4, p3
    if p1.x > p3.x:
        p1, p3 = p3, p1
        p2, p4 = p4, p2    

    # if line is parallel to Y axis then its slope will be infinity which is a valid value
    x_diff = p2.x - p1.x
    slope1 = math.inf if x_diff == 0 else (p2.y - p1.y) / x_diff
    intercept1 = p1.x if slope1 == math.inf else p1.y - slope1 * p1.x

    x_diff = p4.x - p3.x
    slope2 = math.inf if x_diff == 0 else (p4.y - p3.y) / x_diff
    intercept2 = p3.x if slope2 == math.inf else p3.y - slope2 * p3.x

    # When both the segments are on the same line
    if slope1 == slope2:
        if intercept1 == intercept2 and p3.x >= p1.x and p3.x <= p2.x:
            return p3
        else:
            return "Lines are Parllel!"
    
    intersection_x = (intercept2 - intercept1 ) / (slope1 - slope2)
    intersection_y = slope1 * intersection_x + intercept1
    result = Point(intersection_x, intersection_y)

    if result.x >= p1.x and result.x <= p2.x and \
        result.y >= p1.y and result.y <= p2.y:
        return result
    else:
        return "Intersection is outside the given segments!"

if __name__ == "__main1__":
    p1 = Point(1,1)
    p2 = Point(2,2)
    p3 = Point(1,2)
    p4 = Point(2,1)
    print(get_intersection_point(p1, p2, p3, p4))   # Happy path
    print(get_intersection_point(p1, p3, p2, p4))   # Parallel Line
    print(get_intersection_point(p1, p3, p2, Point(5,7))) # Intersection outside the segments
    print(get_intersection_point(p1, p2, Point(1.5,1.5), Point(3,3))) # Same Line


# Problem 16.4: Tic Tac Win: Design an algorithm to figure out if someone has won a game of tic-tac-toe.
'''
    Algorithm:
        1. If we are designing the function just for a 3x3 board then we can simply check the entire board
        without having any performance issues.
        2. If we need to write the function for a NxN board then we need to improve the logic little bit so
        that function will not check the entire board.
        3. One way is to save the last move and we can pass the row and column number of the last move to 
        win function. By this, we just need to check the row, column and diagonal of that row/column to determine
        if someone has won the game or not.
        4. Board cell can have 3 values 0/1/2 and we can always dedicate 1 to player 1 and 2 to player 2.
        Our Win function will return 3 possible values 0/1/2 to make the output clear to understand.
        5. Only tricky part is to implement the diagonal check as we need to check for both the diagonals.
'''
def tic_tac_toe_win(board, row, col):
    def has_row_won(board, row):
        for c in range(1, len(board)):
            if board[row][c] != board[row][0]:
                return False
        return True
    
    def has_col_won(board, col):
        for r in range(1, len(board)):
            if board[r][col] != board[0][col]:
                return False
        return True

    def has_diag_won(board, direction):
        row = 0
        col = 0 if direction == 1 else len(board) - 1
        value = board[row][col]
        for _ in range(len(board)):
            if board[row][col] != value:
                return False
            col += direction
            row += 1
        return True
          
    if board[row][col] == 0: return 0

    if has_row_won(board, row) or has_col_won(board, col):
        return board[row][col]
    if row == col and has_diag_won(board, 1):
        return board[row][col]
    if row == len(board) - col - 1 and has_diag_won(board, -1):
        return board[row][col]
    return 0
        
if __name__ == "__main1__":
    board1 = [
        [1,2,0],
        [2,1,0],
        [0,2,1]
    ]
    print(tic_tac_toe_win(board1, 1, 1))

    board2 = [
        [1,2,2],
        [2,2,0],
        [2,1,0]
    ]
    print(tic_tac_toe_win(board2, 1, 1))


# Problem 16.5: Factorial zeros: Write an algorithm which computes the number of trailing zeros in n factorial.
'''
    Algorithm:
        1. Each 5 or multiple of 5 will contribute a 0.
        2. Each 25 will contribute two 0s [25 * 4].
        3. Each 125 will contribute three 0s [125 * 8].
        4. Each 625 will contribute four 0s [625 * 16].
        5. Implement the pattern in a loop and we will get the exact 0s.
'''
def count_zeros_in_factorial(n):
    if n < 0: return -1
    count = 0
    i = 5
    while n // i > 0:
        count = count + (n // i)
        i = i * 5
    return count 

if __name__ == "__main1__":
    print(count_zeros_in_factorial(25))

# Problem 16.6: Smallest Difference: Given two arrays of integers, compute the pair of values (one value in each
# array) with the smallest (non-negative) difference. Return the difference.
# EXAMPLE
# Input: {1, 3, 15, 11, 2}, {23, 127, 235, 19, 8}
# Output: 3. That is, the pair (11, 8).
'''
    Algorithm:
        1. 3 Approaches possible: Brute Force with O(N^2)
        2. DP with formula dp[i][j] = min(diff, dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) with O(N^2).
        3. Optimized one with O(mlogm + nlogn). 
            a. Sort both the arrays. Take the diff of 1st elements in the sorted arrays, move 1st array if its element
            was smaller else move 2nd array.
            b. Most optimized solution for this problem.
'''
def smallest_difference_brute_force(arr1, arr2):
    if arr1 is None or arr2 is None or len(arr1) == 0 or len(arr2) == 0:
        return
    min = 99999
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            diff = abs(arr1[i] - arr2[j])
            if diff < min:
                min = diff
                pair = (arr1[i], arr2[j])
    print(f'Min Diff is: {min} with Pair: {pair}')

def smallest_difference_dp(arr1, arr2):
    if arr1 is None or arr2 is None or len(arr1) == 0 or len(arr2) == 0:
        return
    m = len(arr1)
    n = len(arr2)
    dp = [[99999 for _ in range(n+1)] for _ in range(m+1)]
    
    least = dp[0][0]    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diff = abs(arr1[i-1] - arr2[j-1])
            dp[i][j] = min(diff, dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
            if dp[i][j] < least:
                least = dp[i][j]
                pair = (arr1[i-1], arr2[j-1])

    print(f'Min Diff is: {dp[m][n]} with Pair: {pair}')

def smallest_difference_sorting(arr1, arr2):
    if arr1 is None or arr2 is None or len(arr1) == 0 or len(arr2) == 0:
        return
    arr1.sort()
    arr2.sort()
    least = 99999
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        diff = abs(arr1[i] - arr2[j])
        if diff < least:
            least = diff
            pair = (arr1[i], arr2[j])
        if arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1
    print(f'Min Diff is: {least} with Pair: {pair}')    

if __name__ == "__main1__":
    arr1 = [1, 3, 15, 11, 2]
    arr2 = [23, 127, 235, 19, 8]
    smallest_difference_brute_force(arr1, arr2)
    smallest_difference_dp(arr1, arr2)
    smallest_difference_sorting(arr1, arr2)


# Problem 16.7: Write a method that finds the maximum of two numbers. You should not use if-else
# or any other comparison operator.
'''
    Algorithm:
        1. Let k = sign(a-b) and q = sign(b-a).
        2. max = k * a + q * b.
        3. To get the sign, we just need to take the MSB of the difference. We can use bitstring.BitArray for that.
        4. BitArray can convert an integer into the Array of Bits and it starts MSB with index 0.
        5. Below code implements this approach.        
'''
def get_max(a, b):
    if a == b: return a
    k = BitArray(int=a-b, length=32)
    q = BitArray(int=b-a, length=32)
    # Sign will be the MSB of the Integer and if its a positive then our value should be 1 so we are using ^1.
    k = int(k[0]) ^ 1
    q = int(q[0]) ^ 1
    return a * k + b * q

if __name__ == "__main1__":
    print(get_max(-4,6))
    print(get_max(-4,-6))
    print(get_max(4,-6))
    print(get_max(4,6))


# Problem 16.8: English Int: Given any integer, print an English phrase that describes the integer (e.g., 
# "One Thousand, Two Hundred Thirty Four").
'''
    Algorithm:
        1. The key to this problem is the amount of code and edge cases.
'''
def english_int(num):
    def convert_chunk(chunk):
        chunk_value = []
        # Hundreds part
        if chunk >= 100:
            chunk_value.append(smalls[chunk // 100] + " " + hundred)
            chunk = chunk % 100
        # Tens part
        if chunk >= 10 and chunk < 20:
            chunk_value.append(smalls[chunk])
            chunk = chunk % 20
        elif chunk >= 20:
            chunk_value.append(tens[chunk // 10])
            chunk = chunk % 10
        # Ones part
        if chunk > 0 and chunk <= 9:
            chunk_value.append(smalls[chunk])
        return ' '.join(chunk_value)

    negative = "Negative"
    smalls = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
    'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Tweleve', 'Thirteen',
    'Fourteen', 'Fifteen', 'Sixtenn', 'Seventeen', 'Eighteen', 'Ninteen']
    tens = ["", "", 'Twenty', 'Thirty', 'Fourty', 'Fifty',
    'Sixty', 'Seventy', 'Eighty', 'Ninty']
    hundred = "Hundred"
    bigs = ["", 'Thousand', 'Million', 'Billion']

    if num == 0: return smalls[num]
    elif num < 0: return negative + " " + english_int(-1 * num)

    result = deque()
    chunk_count = 0
    while num > 0:
        if num % 1000 != 0:
            value = convert_chunk(num % 1000) + " " + bigs[chunk_count]
            result.appendleft(value)
        num = num // 1000
        chunk_count += 1
    return ', '.join(result)

if __name__ == "__main1__":
    print(english_int(1233123119))
    

# Problem 16.9: Operations: Write methods to implement the multiply, subtract, and divide operations for integers.
# The results of all of these are integers. Use only the add operator.
'''
    Algorithm:
        1. Only addition is allowed.
        2. Subtraction is a + -b. So, we need to come up with a method to negate b.
        3. Naive approach to negate b is by adding it to 1/-1 b times based on the sign of b. Time Complexity
        of this approach will be O(b).
        4. Optimized approach for negating a number is to double the pointer in every iteration till the time 
        num + delta is changing the sign. Once sign is changing, set delta to 1 and repeat the process.
        Calculating the Time Complexity of this method is bit complex but on a high level, we can say that it
        is more than O(log b).
        5. Multiply can be done by adding a to itself for b times.
        6. Division is like x = a/b so a = bx. So, we can get x if we keep adding b to itself till b becomes more
        than a.
'''
def operations(a, b):
    def negate_naive(a):
        # Time Complexity: O(a)
        result = 0
        new_sign = 1 if a < 0 else -1
        while a != 0:
            a += new_sign
            result += new_sign
        return result
    
    def negate_optimized(a):
        # Time Complexity: O((log a)^2)
        # Time taken by delta to reach a/2: log a
        # Time taken by delta to reach a/4 from a/2: log a/2 and so on...
        # Time complexity: O(log a) + O(log a/2) + O(log a/4)
        # which will be: O(log a) + O(log a) - log2 + O(log a) - log4 ...
        # On a very high level Time Complexity is: > O(log a)
        result = 0
        new_sign = 1 if a < 0 else -1
        delta = new_sign
        while a != 0:
            sign_changing = (a + delta > 0) == (delta > 0)
            if a + delta != 0 and sign_changing:
                delta = new_sign
            result += delta
            a += delta
            delta += delta
        return result

    def subtract(a, b):
        return a + negate_optimized(b)

    def multiply(a, b):
        abs_a = negate_optimized(a) if a < 0 else a
        abs_b = negate_optimized(b) if b < 0 else b
        if abs_a < abs_b:
            return multiply(b, a)
        result = 0
        i = abs_b
        while i > 0:
            result += a
            i = subtract(i, 1)
        if b < 0:
            return negate_optimized(result)
        return result

    def divide(a, b):
        if b == 0:
            raise ValueError("Divide by Zero!")
        abs_a = negate_optimized(a) if a < 0 else a        
        abs_b = negate_optimized(b) if b < 0 else b
        result = 0
        product = abs_b
        while product + abs_b <= abs_a:
            product += abs_b
            result += 1
        if (a > 0 and b < 0) or (a < 0 and b > 0):
            return negate_optimized(result)
        return result

    print(f'{a} - {b} = {subtract(a, b)}')
    print(f'{a} * {b} = {multiply(a, b)}')
    print(f'{a} / {b} = {divide(a, b)}')
    print()

if __name__ == "__main1__":
    operations(2, 4)
    operations(-2, -4)
    operations(2, -4)
    operations(-4, 2)


# Problem 16.10: Living People: Given a list of people with their birth and death years, implement a method to
# compute the year with the most number of people alive. You may assume that all people were born
# between 1900 and 2000 (inclusive). If a person was alive during any portion of that year, they should
# be included in that year's count. For example, Person (birth= 1908, death= 1909) is included in the
# counts for both 1908 and 1909.
'''
    Algorithm:
        1. Given details are important to solve this problem.
        2. All people were born in between 1900 and 2000 so our values are in a fixed range.
        3. In Brute Force approach, we will check each and every year and compare the values against 
        birth and death years of all the people to get the living count for that year.
    
    Optimized Approach:
        1. We can create 2 lists, one for birth years and another for death years and then sort them.
        2. Since we only need to check all the birth years in order to find out the maximum living people.
        3. We will keep jumping between years in birth years and if we encounter any death year in between, 
        we will decrement the living count.
        4. Comparison of index values are quite important here.
'''
def living_people_brute_force(people_list, start, end):
    max_living = 0
    max_living_year = 0
    for r in range(start, end + 1):
        living = 0
        for person in people_list:            
            if person[0] <= r and person[1] >= r:
                living += 1
        if living > max_living:
            max_living = living
            max_living_year = r
    print(f'Maximum Living People were {max_living} in year {max_living_year}')

def living_people_sorting(people_list):
    birth_years = [x[0] for x in people_list]
    death_years = [x[1] for x in people_list]
    birth_years.sort()
    death_years.sort()
    i = j = 0
    max_living = max_living_year = living = 0
    while i < len(birth_years):
        if birth_years[i] <= death_years[j]:
            living += 1
            if living > max_living:
                max_living = living
                max_living_year = birth_years[i]
            i += 1            
        elif birth_years[i] > death_years[j]:
            living -= 1
            j += 1

    print(f'Maximum Living People were {max_living} in year {max_living_year}')

if __name__ == "__main1__":    
    people_list = [
        # (1901, 1915), (1910, 1972), (1910, 1982), (1912, 1990), (1913, 1994),
        # (1920, 1998), (1923, 1998), (1975, 1998), (1983, 1998), (1990, 1999)
        (1901, 1915), (1901, 1915), (1901, 1915), (1901, 1915), (1901, 1915),
        (1901, 1915), (1901, 1915), (1901, 1915), (1901, 1915), (1901, 1915)
    ]
    living_people_brute_force(people_list, 1900, 2000)
    living_people_sorting(people_list)


# Problem 16.11: Diving Board: You are building a diving board by placing a bunch of planks of wood end-to-end.
# There are two types of planks, one of length shorter and one of length longer. You must use
# exactly K planks of wood. Write a method to generate all possible lengths for the diving board.
'''
    Algorithm:
        1. One solution can be made with DP. L(k) = L(k-1) + plank_length. Base Case: L(0) = plank_length.
        2. For each k, we need to pick both type of planks to get all possible lengths.
        3. We need to store all the lenghts in Set to avoid duplicates. We can use memoization as well to save time.
        4. Memoization is bit different here. Key will be running length + k. If its in the set then we just return as
        this combination has already been counted else we will add the key in the cache set.
        5. Time Complexity of DP is O(2^K).
    
    Optimal and Best Solution:
        1. Final Result will be a list of following lengths:
            Choose (0 type A plank, k type B plank), (1 type A plank, k-1 type B plank),
            (2 type A plank, k-2 type B plank) .... and (k type A plank, 0 type B plank).
        2. Any path with same number of each type of planks will have the same length. Since we can have
        at most k planks of each type, there are only K different sums we can make.
'''
def diving_board_dp(shorter, longer, k):
    def diving_board_util(shorter, longer, k , result, length):
        if k == 0:
            result.add(length)
            return
        diving_board_util(shorter, longer, k-1, result, length + shorter)
        diving_board_util(shorter, longer, k-1, result, length + longer)

    if shorter == longer:
        print(shorter * k)
        return
    result = set()
    diving_board_util(shorter, longer, k, result, 0)
    for r in result:
        print(r)
    print()

def diving_board_dp_memoization(shorter, longer, k):
    def diving_board_dp_memoization_util(shorter, longer, k , result, length, cache):
        if k == 0:
            result.add(length)
            return
        key = str(length) + ' ' + str(k)
        if key in cache: return
        diving_board_dp_memoization_util(shorter, longer, k-1, result, length + shorter, cache)
        diving_board_dp_memoization_util(shorter, longer, k-1, result, length + longer, cache)
        cache.add(key)
    
    if shorter == longer:
        print(shorter * k)
        return
    result = set()
    cache = set()
    diving_board_dp_memoization_util(shorter, longer, k, result, 0, cache)
    for r in result:
        print(r)
    print()


def diving_board_optimized(shorter, longer, k):
    if shorter == longer:
        print(shorter * k)
        return
    result = []
    for i in range(k+1):
        result.append(longer * i + shorter * (k - i))
    for r in result:
        print(r)

if __name__ == "__main1__":
    diving_board_dp(2, 4, 4)
    diving_board_dp_memoization(2, 4, 4)
    diving_board_optimized(2,4,4)


# Problem 16.12: XML Encoding: Since XML is very verbose, you are given a way of encoding it where each tag gets
# mapped to a pre-defined integer value. The language/grammar is as follows:
# Element --> Tag Attributes END Children END
# Attribute --> Tag Value
# END --> 0
# Tag --> some predefined mapping to int
# Value --> string value
# For example, the following XML might be converted into the compressed string below (assuming a
# mapping of family -> 1, person ->2, firstName -> 3, lastName -> 4, state -> 5).
# <family lastName="McDowell" state="CA">
#   <person firstName="Gayle">Some Message</ person>
# </ family>
# Becomes:
# 1 4 McDowell 5 CA 0 2 3 Gayle 0 Some Message 0 0
# Write code to print the encoded version of an XML element (passed in as Element and Attribute objects).
'''
    Algorithm:
        1. Since input is an Element object which has child Elements and Attribut objects, we can 
        process the XML in a Tree like structure.
'''
class Element(object):
    def __init__(self, code, value=None):
        self.code = code
        self.value = value
        self.attributes = []
        self.children = []
    def set_attributes(self, attributes):
        for attribute in attributes:
            self.attributes.append(attribute)
    def set_children(self, children):
        for element in children:
            self.children.append(element)

class Attribute(object):
    def __init__(self, tag_code, value=None):
        self.tag_code = tag_code
        self.value = value

def xml_encoding(element):
    def encode_element(element, result):
        result.append(element.code)
        for attribute in element.attributes:
            encode_attribute(attribute, result)
        result.append('0')
        if element.value is not None and element.value != "":
            result.append(element.value)
        else:
            for element in element.children:
                encode_element(element, result)
        result.append('0')

    def encode_attribute(attribute, result):
        result.append(attribute.tag_code)
        result.append(attribute.value)
    
    result = []
    encode_element(element, result)
    print(' '.join(result))

if __name__ == "__main1__":
    family = Element('1')
    lastName = Attribute('4', 'McDowell')
    state = Attribute('5', 'CA')
    family.set_attributes([lastName, state])
    person = Element('2', 'Some Message')
    firstName = Attribute('3', 'Gayle')
    person.set_attributes([firstName])
    family.set_children([person])
    xml_encoding(family)


# Problem 16.13: Bisect Squares: Given two squares on a two-dimensional plane, find a line that would cut these two
# squares in half. Assume that the top and the bottom sides of the square run parallel to the x-axis.
'''
    Algorithm:
        1. Solution is known to every one for this problem.
        2. Line connecting the middle point of both the squares will divide them in half.
        3. We need to code this problem by keeping special cases in mind.
            A. 3 variations are possible to define the line: 
                i. Just the line slope and its intercept.
                ii. Just any 2 points on the line.
                iii. Or 2 points that matches with start and ends of the square edges.
                3rd variation will make the problem bit more complicated.
            B. If both mid points have same x coordinates then line slope will be a division by 0.
            C. Find out which direction line will go that is from Square 1 to Square 2 or from Square 2 to Square 1.
            This can be known by finding out which square has its left point closer to the origin.
            D. Based on the line direction, we need to find out the edges of both the squares to get the intersection
            points.
            E. If both the squares have same middle point then find out the bigger Square to get the line 
            segments.
'''


# Problem 16.14: Given a two-dimensional graph with points on it, find a line which passes the most
# number of points.


# Problem 16.15: Master Mind: The G ame of Master Mind is played as follows:
# The computer has four slots, and each slot will contain a ball that is red (R), yellow (Y), green (G) or
# blue (B). For example, the computer might have RGGB (Slot #1 is red, Slots #2 and #3 are green, Slot
# #4 is blue).
# You, the user, are trying to guess the solution. You might, for example, guess YRGB.
# When you guess the correct color for the correct slot, you get a "hit:' If you guess a color that exists
# but is in the wrong slot, you get a "pseudo-hit:' Note that a slot that is a hit can never count as a
# pseudo-hit.
# For example, if the actual solution is RGBY and you guess GGRR , you have one hit and one pseudohit
# Write a method that, given a guess and a solution, returns the number of hits and pseudo-hits.
'''
    Algorithm:
        1. Algorithm to count the values is very straight forward but logic can be written in a neat way
        if we maintain a Hash Table that will keep counts of colors that can be counter in Pseudo-hits 
        calculation.
'''
def master_mind(actual, guess):
    if len(actual) != len(guess): return None
    for r in guess:
        if r not in ['R','Y','G','B']:
            return None
    hits = 0
    pseudo_hits = 0
    frequency = {
        'R': 0, 'Y': 0, 'G': 0, 'B': 0
    }
    # Counting hits and building frequency table
    for i in range(len(guess)):
        if guess[i] == actual[i]:
            hits += 1
        else:
            frequency[actual[i]] += 1
    # Counting pseudo-hits 
    for i in range(len(guess)):
        f = frequency[guess[i]]
        if f > 0 and guess[i] != actual[i]:
            pseudo_hits += 1
            frequency[guess[i]] -= 1
    print(f'Hits: {hits}, Pseudo-hits: {pseudo_hits}')

if __name__ == "__main1__":
    master_mind("RGBY", "GGRR")
    master_mind("RGBR", "GGRR")
        

# Problem 16.16: Sub Sort: Given an array of integers, write a method to find indices m and n such that if you sorted
# elements m through n , the entire array would be sorted. Minimize n - m (that is, find the smallest
# such sequence).
# EXAMPLE
# Input: 1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19
# Output: (3, 9)
'''
    Algorithm:
        1. This algorithm is easy to think through but difficult to implement due to a lot of index movements.
        2. Approach is simple:
            Find out the left part which is correctly sorted.
            Find out the right part which is correctly sorted.
        3. Now, array has been divided in 3 parts: left sorted, middle unsorted and right sorted.
        4. This is the most crucial part of the logic.
            If unsorted part has any element which is less than max of left part or more than min of
            right part then we need to re-adjust the left and right parts (reduce their length basically).
            
            So, we need to find the min and max element index in the unsorted part but can we have any starting
            values. Yes we can, max value should be more than left max and min value should be more than
            right min.

            We can find index of min and max element without any starting point as well but above approach 
            will make the code slightly compact to find the correct index value.
            
        5. Now, we slide the left part to further left till index value is less than or equal to the min_index.
        6. Slide the right part to further right till index value is more than or equal to the max_index.
    
    Easier Algorithm:
        1. Sort the array and store it in a temp array.
        2. Compare the start of both the arrays and first non matching index is the left index.
        3. Compare the end of both the arrays in reverse direction and first non matching index is the right index.
        4. But this is a sorting problem and if we sort the array already to come up with the solution then whole
        purpose of the solution will be defeated so its not at all an accepted solution.
'''
def sub_sort(arr):
    def get_left_end(arr, n):
        for i in range(1, n):
            if arr[i] < arr[i-1]:
                return i - 1
        return n - 1
    
    def get_right_start(arr, n):
        for i in reversed(range(n-1)):
            if arr[i] > arr[i+1]:
                return i + 1
        return 0

    def slide_left(arr, min_index, left_end):
        value = arr[min_index]
        for i in reversed(range(left_end)):
            if arr[i] <= value:
                return i + 1
        return 0

    def slide_right(arr, max_index, right_start):
        value = arr[max_index]
        for i in range(right_start, len(arr)):
            if arr[i] >= value:
                return i - 1
        return len(arr) - 1
    
    if arr is None or len(arr) == 0: return
    n = len(arr)
    # End of Left Sorted Part    
    left_end = get_left_end(arr, n)    
    # Array is already sorted
    if left_end == n - 1: return []

    # Start of Right Sorted Part    
    right_start = get_right_start(arr, n)

    max_index = left_end
    min_index = right_start
    for i in range(left_end + 1, right_start):
        if arr[i] < arr[min_index]: min_index = i
        if arr[i] > arr[max_index]: max_index = i

    # Sliding Left Array
    left_index = slide_left(arr, min_index, left_end)

    # Sliding Right Array
    right_index = slide_right(arr, max_index, right_start)

    print(f'Index Range is: [{left_index}, {right_index}]. Unsorted Part is: {arr[left_index: right_index + 1]}')

def sub_sort_by_sorting(arr):
    if arr is None or len(arr) == 0: return
    temp = sorted(arr)
    left_index = 0
    for i in range(len(arr)):
        if temp[i] != arr[i]:
            left_index = i
            break
    if left_index == len(arr) - 1: return []
    right_index = len(arr) - 1
    for i in reversed(range(len(arr))):
        if temp[i] != arr[i]:
            right_index = i
            break
    print(f'Index Range is: [{left_index}, {right_index}]. Unsorted Part is: {arr[left_index: right_index + 1]}')

if __name__ == "__main1__":
    sub_sort([1, 2, 4, 7, 10, 11, 27, 12, 6, 7, 16, 18, 19])
    sub_sort_by_sorting([1, 2, 4, 7, 10, 11, 27, 12, 6, 7, 16, 18, 19])


# Problem 16.17: Contiguous Sequence: You are given an array of integers (both positive and negative). Find the
# contiguous sequence with the largest sum. Return the sum.
# EXAMPLE
# Input: 2, -8, 3, -2, 4, -10
# Output: 5 ( i. e. , { 3, -2, 4} )
'''
    Algorithm:
        1. Logic behind this algorithm is this: If we add a positive and negative number and their sum is less than 0
        then can never be part of the result. An array can be assumed as a group of subsequences of plus and 
        minus numbers.
        2. Take 2 variables, sum and max_sum. Start both with 0.
        3. If sum is less than 0, set it back to 0. If sum is more than max_sum then set max_sum to sum.
        4. In each iteration, add array element to sum and check. 
        5. Boundary case: What should be the result when array has all negative numbers:
            i. Throw an error
            ii. return 0. This approach will return this.
            iii. return a pre-defined big negative number.       
            All 3 are valid solutions.
'''
def contiguous_sequence(arr):
    if arr is None or len(arr) == 0: return None
    max_sum = 0
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
        if sum > max_sum:
            max_sum = sum
        if sum < 0:
            sum = 0
    return max_sum

if __name__ == "__main1__":
    print(contiguous_sequence([2, -8, 3, -2, 0, -10]))


# Problem 16.18 Pattern Matching: You are given two strings, pattern and value. The pattern string consists of
# just the letters a and b, describing a pattern within a string. For example, the string catcatgocatgo
# matches the pattern aabab (where cat is a and go is b). It also matches patterns like a, ab, and b.
# Write a method to determine if value matches pattern.
'''
    Algorithm:
        1. This problem's Brute Force algorithm is also slightly tricky so ignore brute force implementation.
    
    Optimized and interesting algorithm:
        1. This implementation is based on number of a and b in the pattern.
        2. If we know the total number of a then we can find out the max length of a.
        3. Once we find out the max length of a, we can try all possible lengths of a and respective possible
        lengths for b to match the pattern.
        4. This is how the logic works:
            a. find main char, alt char, main count, alt count, index of first alt and max size of main char.
            b. For each possible size of main char, try matching the pattern with value.
            c. How the matching logic works:
                a. Based on the main count and main size, find the remaining length of value.
                b. Based on the rem length and alt count, find the alt size (will be 0 if rem length is 0).
                c. Calculate the index where alt will appear first.
                d. During matching, we will start comparing the pattern from its 1st char.
                e. if current char is same as main char then we know the exact range of indexes where
                chars in value should match.
                    if char is main: (start till main_size) should match (value index counter till main size)
                    if char is alt: (first alt char index till alt size) should match (value index counter till alt size).
'''
def pattern_matching_brute_force(pattern, value):
    def build_pattern(pattern, main, alt):
        result = ''
        first = pattern[0]
        for r in pattern:
            if r == first:
                result += main
            else:
                result += alt
        return result

    if len(pattern) == 0: return len(value) == 0
    n = len(value)
    for main_size in range(n):
        main = value[0:main_size]
        for alt_start in range(main_size, n+1):
            for alt_end in range(alt_start, n+1):
                alt = value[alt_start: alt_end]
                temp = build_pattern(pattern, main, alt)
                if temp == value:
                    return 'True' + ' : ' + main + ' : ' + alt
    return False

def pattern_matching(pattern, value):
    def matches(pattern, value, main_size, alt_size, first_alt):
        def is_equal(s1, offset1, offset2, size):
            for i in range(size):
                if s1[offset1 + i] != s1[offset2 + i]:
                    return False
            return True

        offset2 = main_size
        for i in range(1, len(pattern)):
            size = main_size if pattern[i] == pattern[0] else alt_size
            offset = 0 if pattern[i] == pattern[0] else first_alt
            if not is_equal(value, offset, offset2, size):
                return False
            offset2 += size
        print(f'a = {value[0:main_size]}, b = {value[first_alt:first_alt+alt_size]}')
        return True
    
    if len(pattern) == 0: return len(value) == 0

    main_char = pattern[0]
    alt_char = 'a' if main_char == 'b' else 'b'
    n = len(value)
    main_count = Counter(pattern)[main_char]
    alt_count = len(pattern) - main_count
    first_alt = pattern.index(alt_char)
    max_main_size = n // main_count

    for main_size in range(max_main_size):
        rem_length = n - main_size * main_count
        if alt_count == 0 or rem_length % alt_count == 0:
            alt_index = first_alt * main_size
            alt_size = 0 if rem_length == 0 else rem_length // alt_count
            if matches(pattern, value, main_size, alt_size, alt_index):
                return True
    return False

if __name__ == "__main1__":
    print(pattern_matching_brute_force("aabab", "catcatgocatgo"))
    print(pattern_matching("aabab", "catcatgocatgo"))


# Problem 16.19: Pond Sizes: You have an integer matrix representing a plot of land, where the value at that location
# represents the height above sea level. A value of zero indicates water. A pond is a region of water
# connected vertically, horizontally, or diagonally. The size of the pond is the total number of
# connected water cells. Write a method to compute the sizes of all ponds in the matrix.
# EXAMPLE
# Input:
# 0 2 1 0
# 0 1 0 1
# 1 1 0 1
# 0 1 0 1
# Output: 2, 4, 1 (in any order)
'''
    Algorithm:
        1. Its a Matrix DFS/BFS traversal problem. 
        2. Each time we get a cell with value 0, we start the pond with size 1 and check its all neighbors to
        find if pond size can be bigger or not.
        3. We need to maintain a buffer matrix as well to save the visit status of a cell as we only need to
        visit a cell once.
        4. Time complexity: O(mn)
'''
def pond_sizes(matrix):
    def compute_size(matrix, visited, r, c):
        if r < 0 or c < 0 or r >= len(matrix) or c >= len(matrix[0]) \
            or visited[r][c] or matrix[r][c] != 0:
            return 0
        size = 1        
        visited[r][c] = True
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                # this loop will run 9 times including current cell as well but it wont affect anything
                # as we have already marked its visited status to True.
                size += compute_size(matrix, visited, r + dr, c + dc)
        return size
    
    m = len(matrix)
    n = len(matrix[0])    
    visited = [[False for x in range(n)] for x in range(m)]
    result = []
    for r in range(m):
        for c in range(n):
            size = compute_size(matrix, visited, r, c)
            if size > 0:
                result.append(size)

    print(', '.join([str(x) for x in result]))

if __name__ == "__main1__":
    matrix = [
        [0,2,1,0],
        [0,1,0,1],
        [1,1,0,1],
        [0,1,0,1]
    ]
    pond_sizes(matrix)


# Problem 16.20: T9: On old cell phones, users typed on a numeric keypad and the phone would provide a list of words
# that matched these numbers. Each digit mapped to a set of O - 4 letters. Implement an algorithm
# to return a list of matching words, given a sequence of digits. You are provided a list of valid words
# (provided in whatever data structure you'd like). The mapping is shown in the diagram below:
#             1           2 (abc)     3 (def)
#             4 (ghi)     5 (jkl)     6 (mno)
#             7 (pqrs)    8 (tuv)     9 (wxyz)
# EXAMPLE:
# Input: 8733
# Output: tree, used
'''
    Algorithm:
        Brute Force:
        1. Brute Force can be applied via making recursive calls.
        2. For each input number, we need to make up to 4 recursive calls and 4 recursive calls for each 1 of them.
        So if input length is N then time complexity of Brute Force will be O(4^N) which is very slow.

        Prefix Tree or Trie:
        1. This is a perfect case for using Trie.
        2. We will store all the valid list of words in the Trie and then search the prefix in Trie which will 
        make the algorithm faster a lot.
        3. Exact time complexity will be difficult to compute.

        Using Hash Table (fastest):
        1. Create a Hash Table that maps a number to the list of words.
        2. Add all the valid words into this Hash Table by converting them into their T9 digits. Ex. APPLE will be
        27753.
        3. Once hash table is ready, we just need to do a look up for given input number in the Hash Table and
        we will get our valid list of words.
        4. If Hash Table creation is not taken into account then its a O(1) time complexity.
'''
def valid_t9_words_brute_force(input, word_dict):
    def get_valid_words(input, index, prefix, word_dict, result):
        if index == len(input):
            if prefix in word_dict:
                result.append(prefix)
            return

        global mapping
        digit = input[index]
        letters = mapping[digit]
        if letters is None: return
        for letter in letters:
            get_valid_words(input, index + 1, prefix + letter, word_dict, result)
    
    global mapping
    mapping = {
        '0': None, '1': None, '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno',
        '7': 'pqrs', '8': 'tuv', '9': 'wxyz' 
    }
    result = []
    get_valid_words(str(input), 0, '', word_dict, result)
    print(', '.join(result))


class TrieNode(object):
    def __init__(self):
        self.children = [None]*26
        self.isLeaf = False

class Trie(object):
    def __init__(self):
        # Root will be a None TrieNode
        self.root = TrieNode()

    def insert(self, word):
        # If word is not present then insert the word into Trie.
        # If word is prefix of Trie then just mark the leaf node to make it a end of word.
        node = self.root
        length = len(word)
        for r in range(length):
            index = ord(word[r]) - ord('a')
            if node.children[index] is None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isLeaf = True

def valid_t9_words_trie(input, trie):
    def get_valid_words_trie(input, index, prefix, trie_node, result):
        if index == len(input):
            if trie_node.isLeaf:
                result.append(prefix)
            return

        global mapping
        digit = input[index]
        letters = mapping[digit]
        if letters is None: return
        
        for letter in letters:
            trie_index = ord(letter) - ord('a')
            node = trie_node.children[trie_index]
            if node is not None:
                get_valid_words_trie(input, index + 1, prefix + letter, node, result)
    
    global mapping
    mapping = {
        '0': None, '1': None, '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno',
        '7': 'pqrs', '8': 'tuv', '9': 'wxyz' 
    }
    result = []
    get_valid_words_trie(str(input), 0, '', trie.root, result)
    print(', '.join(result))

if __name__ == "__main1__":
    word_dict = ['tree','used','sued','used','usage']
    # valid_t9_words_brute_force(8733, word_dict)

    trie = Trie()
    for word in ['tree','used','sued','used','usage']:
        trie.insert(word)
    valid_t9_words_trie(8733, trie)


# Problem 16.21: Sum Swap: Given two arrays of integers, find a pair of values (one value from each array) that you
# can swap to give the two arrays the same sum.
# EXAMPLE
# lnput:{4, 1, 2, 1, 1, 2} and {3, 6, 3, 3}
# Output: {1, 3}
'''
    Analysis:
        If we are swapping a and b from arr1 and arr2. Then this will hold true:
            sum1 - a + b = sum2 + a - b
            2a - 2b = sum1 - sum2
            a - b = (sum1 - sum2) / 2
        Since values are integers only, then difference of 2 sums have to be an even number else pair will 
        not be found.
'''
def sum_swap_brute_force(arr1, arr2):
    if len(arr1) == 0 or len(arr2) == 0: return
    sum1 = sum(arr1)
    sum2 = sum(arr2)
    for i in arr1:
        for j in arr2:
            new_sum1 = sum1 - i + j
            new_sum2 = sum2 + i - j
            if new_sum1 == new_sum2:
                print(f'{i}, {j}')
                return
    print("No Pair Found")

def sum_swap_sorting(arr1, arr2):
    def get_target(arr1, arr2):
        sum1 = sum(arr1)
        sum2 = sum(arr2)
        if (sum1 - sum2) % 2 != 0: return None
        return (sum1 - sum2) // 2

    def find_pair(arr1, arr2, target):
        i = j = 0
        while i < len(arr1) and j < len(arr2):
            diff = arr1[i] - arr2[j]
            if diff == target:
                print(f'{arr1[i]}, {arr2[j]}')
                return
            if diff > target:
                j += 1
            else:
                i += 1
        print("No Pair Found")
    
    if len(arr1) == 0 or len(arr2) == 0: return
    arr1.sort()
    arr2.sort()
    target = get_target(arr1, arr2)
    if target is None: return "No Pair Found"
    find_pair(arr1, arr2, target)

if __name__ == "__main1__":
    arr1 = [4,1,2,1,1,2]
    arr2 = [3,6,3,3]
    sum_swap_brute_force(arr1, arr2)
    sum_swap_sorting(arr1, arr2)    


# Problem 16.22: Langton's Ant: An ant is sitting on an infinite grid of white and black squares. It initially faces right.
# At each step, it does the following:
# (1) At a white square, flip the color of the square, turn 90 degrees right (clockwise), and move forward
# one unit.
# (2) At a black square, flip the color of the square, turn 90 degrees left (counter-clockwise), and move
# forward one unit.
# Write a program to simulate the first K moves that the ant makes and print the final board as a grid.
# Note that you are not provided with the data structure to represent the grid. This is something you
# must design yourself. The only input to your method is K. You should print the final grid and return
# nothing. The method signature might be something like void printKMoves ( int K).
'''
    Algorithm:
        1. Although it may seem "obvious" that we would use a matrix to represent a grid, it's actually easier not to do
        that. All we actually need is a list of the white squares (as well as the ant's location and orientation).
        2. We can do this by using a HashSet of the white squares. If a position is in the hash set, then the square is
        white. Otherwise, it is black.
        3. The one tricky bit is how to print the board. Where do we start printing? Where do we end?
        4. Since we will need to print a grid, we can track what should be top-left and bottom-right corner of the grid.
        Each time the ant moves, we compare the ant's position to the most top-left position and most bottomright
        position, updating them if necessary.
'''
class Position(object):
    def __init__(self, row=None, col=None):
        self.row = row
        self.col = col
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.row == other.row and self.col == other.col
    def clone(self):
        return Position(self.row, self.col)

class Ant(object):
    def __init__(self):
        self.position = Position(0, 0)
        self.dir = 'R'
    def get_turn(self, clockwise):
        turn = ''
        if self.dir == 'L':
            turn = 'U' if clockwise else 'D'
        elif self.dir == 'R':
            turn = 'U' if not clockwise else 'D'
        elif self.dir == 'U':
            turn = 'R' if clockwise else 'L'
        elif self.dir == 'D':
            turn = 'R' if not clockwise else 'L'
        return turn
    def turn(self, clockwise):
        self.dir = self.get_turn(clockwise)
    def move(self):
        if self.dir == 'L':
            self.position.col -= 1
        elif self.dir == 'R':
            self.position.col += 1
        elif self.dir == 'U':
            self.position.row -= 1
        elif self.dir == 'D':
            self.position.row += 1

class Board(object):
    def __init__(self):
        self.ant = Ant()
        self.whites = []
        self.top_left = Position(0, 0)
        self.bottom_right = Position(0, 0)
    def move(self):
        self.ant.turn(self.is_position_white(self.ant.position))
        self.flip(self.ant.position)
        self.ant.move()
        self.adjust_corners(self.ant.position)
    def is_position_white(self, position):
        return position in self.whites
    def is_row_col_white(self, row, col):
        return Position(row, col) in self.whites
    def flip(self, position):
        if position in self.whites:
            self.whites.remove(position)
        else:
            self.whites.append(position.clone())
    def adjust_corners(self, position):
        row = position.row
        col = position.col
        self.top_left.row = min(self.top_left.row, row)
        self.top_left.col = min(self.top_left.col, col)
        self.bottom_right.row = max(self.bottom_right.row, row)
        self.bottom_right.col = max(self.bottom_right.col, col)
    def __repr__(self):
        result = []
        min_row = self.top_left.row
        min_col = self.top_left.col
        max_row = self.bottom_right.row
        max_col = self.bottom_right.col
        for r in range(min_row, max_row):
            for c in range(min_col, max_col):
                # if self.ant.position.row == r and self.ant.position.col == c:
                #     result.append(self.ant.dir)
                if self.is_row_col_white(r, c):
                    result.append("X")
                else:
                    result.append("_")
            result.append("\n")
        result.append("Ant: " + self.ant.dir + "\n")
        return ''.join(result)

def move_langton_ant(K):
    board = Board()
    for _ in range(K):
        board.move()        
    print(board)

if __name__ == "__main1__":
    move_langton_ant(1000)


# Problem 16.23: Rand7 from Rand 5: Implement a method rand7() given rand5(). That is, given a method that
# generates a random number between 0 and 4 (inclusive), write a method that generates a random
# number between 0 and 6 (inclusive).
'''
    Algorithm:
        1. The objective here is to get a series of values which are in range of 7 or more and they must
        be equally likely. Only then we can guarantee of the equal probability.
        2. In below implementation, we are using 2 rand5 calls in each iteration so both the output will give
        us a possible set of 25 values from 0,0 to 4,4 and based on that num will be between 0 to 24 exactly once.
        Hence our function will work correctly.
        3. Key here is to use the fact that we can make any number of calls to rand5.
        4. There is a way to use other multipliers than 5 as well but it will be more complicated.
'''
def rand7_from_rand5():
    def rand5():
        return random.randint(0,4)
    while True:
        num = 5 * rand5() + rand5()
        if num < 21:
            return num % 7

if __name__ == "__main1__":
    result = [0,0,0,0,0,0,0]
    for _ in range(1000):
        result[rand7_from_rand5()] += 1
    for k,v in enumerate(result):
        print(k,v)
        

# Problem 16.24: Pairs with Sum: Design an algorithm to find all pairs of integers within an array which sum to a
# specified value.
'''
    Algorithm:
        Hash Table:
        1. We can create a Counter Hash Table with given array.
        2. For each Counter key, we can search for its complement value which is sum - element.
        3. We need to adjust the counts of element and its complement correctly and add respective 
        number of entries in the result.
        4. This will take O(N) time.

        Sorting and Binary Search:
        1. Sort the array.
        2. For each element, do a binary search for its complement in the remaining array.
        3. It will take O(NlogN) time.
'''
def pairs_with_sum(nums, sum):
    # This will return all the possible pairs (even if both the elements are same)
    result = []
    hash_table = Counter(nums)
    for r in nums:
        if hash_table[r] > 0:
            # Subtracting 1 here for current index otherwise if current and diff value are same and there is
            # only 1 element then it will make a pair out of it. Ex: [0] with sum = 0
            hash_table[r] -= 1
            diff = sum - r
            if diff in hash_table and hash_table[diff] > 0:
                result.append([r, diff])
                hash_table[diff] -= 1 
                # In case we only need to find unique elements then below 2 lines will do that.
                # hash_table[diff] = 0
                # hash_table[r] = 0       
    for r in result:
        print(r)

if __name__ == "__main1__":
    pairs_with_sum([-2,-1,0,0,5,3,5,6,7,9,13,14], 5)


# Problem 16.25: LRU Cache: Design and build a "least recently used" cache, which evicts the least recently used item.
# The cache should map from keys to values (allowing you to insert and retrieve a value associated
# with a particular key) and be initialized with a max size. When it is full, it should evict the least
# recently used item. You can assume the keys are integers and the values are strings.
'''
    Algorithm:
        1. Very useful algorithm.
        2. We need to mix Hash Table and Doubly Linked List to maintain the faster lookup and access order information.
        3. Hash Table will be a dictionary its key will be item key and value will Linked List Node with key and value both.
        4. Linked List node will have both Key and Value in it.
'''
class LRUCache(object):
    class LinkedListNode(object):
        def __init__(self, key, value, next=None, prev=None):
            self.next = next
            self.prev = prev
            self.key = key
            self.value = value
    def __init__(self, max_size):
        self.max_size = max_size
        self.size = 0
        self.hash_table = {}
        self.head = None
        self.tail = None
    def remove_from_list(self, node):
        if node is None: return
        if node.prev: node.prev.next = node.next
        if node.next: node.next.prev = node.prev
        if node == self.head: self.head = node.next
        if node == self.tail: self.tail = node.prev
    def add_in_start(self, node):
        if node is None: return
        if self.head is None:
            self.head = self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
    def get_hash_table_item(self, key):
        if key in self.hash_table:
            return self.hash_table[key]
    def remove_hash_table_item(self, key):
        if key in self.hash_table:
            del self.hash_table[key]

    def get_value(self, key):
        node = self.get_hash_table_item(key)
        if node is None: return
        if node != self.head:
            self.remove_from_list(node)
            self.add_in_start(node)
        return node.value

    def remove_key(self, key):
        node = self.get_hash_table_item(key)
        self.remove_from_list(node)
        self.remove_hash_table_item(key)
        return True

    def set_key(self, key, value):
        self.remove_key(key)
        if len(self.hash_table) == self.max_size and self.tail:
            self.remove_key(self.tail.key)
        node = LRUCache.LinkedListNode(key, value)
        self.add_in_start(node)
        self.hash_table[key] = node
    
    def __repr__(self):
        result = []
        node = self.head
        while node:
            result.append('['+str(node.key)+':'+str(node.value)+']')
            node =  node.next
        return ' -> '.join(result)

if __name__ == "__main1__":
    cache = LRUCache(5)
    for r in range(5):
        cache.set_key(r, 100)
    print(cache)
    cache.remove_key(0)
    print(cache)
    print(cache.get_value(1))
    print(cache)
    cache.set_key(5, 500)
    print(cache)
    cache.set_key(10, 1000)
    print(cache)
    

# Problem 16.26: Calculator: Given an arithmetic equation consisting of positive integers,+,-,* and/ (no parentheses),
# compute the result.
# EXAMPLE
# Input: 2*3+5/6*3+15
# Output: 23.5
'''
    Algorithm:
        1. We can use 2 stacks to calculate an expression.
        2. Whenever we get a number, push it on num stack. For operator, push it on op stack.
        3. Before pushing an operator, check if its priority is less than or equal to the top
        operator of op stack. If yes then pop top 2 nums from num stack and pop top operator
        from op stack, calculate the result and push it to num stack.
        4. Boundary conditions need to be applied carefully here.
'''
def calculator(exp):
    def get_number(exp, start):
        result = ''
        for i in range(start, len(exp)):
            if exp[i] == '.' or exp[i].isnumeric():
                result += exp[i]
            else:
                break
        return result

    def get_op(exp, start):
        if start < len(exp):
            return exp[start]

    def calculate(num1, op, num2):
        if op == '+': return int(num1) + int(num2)
        if op == '-': return int(num1) - int(num2)
        if op == '*': return int(num1) * int(num2)
        if op == '/': return int(num1) / int(num2)

    def collapse(op, num_stack, op_stack):
        priority = {'+': 1, '-': 1, '*': 2, '/': 2, ' ': 0}
        while len(num_stack) >= 2 and len(op_stack) >= 1:
            if priority[op] <= priority[op_stack[len(op_stack) - 1]]:
                num2 = num_stack.pop()
                num1 = num_stack.pop()
                oper = op_stack.pop()
                calc_value = calculate(num1, oper, num2)
                num_stack.append(calc_value)
            else:
                break
    
    num_stack = []
    op_stack = []
    i = 0
    while i < len(exp):
        value = get_number(exp, i)
        num_stack.append(value)

        i += len(value)
        if i >= len(exp): break

        op = get_op(exp, i)
        collapse(op, num_stack, op_stack)
        op_stack.append(op)
        i += 1

    # This case will come when 2 numbers are on stack and 1 operator and loop hit a break (ex: 2/3)
    collapse(' ', num_stack, op_stack)
    if len(num_stack) == 1 and len(op_stack) == 0:
        print(num_stack.pop())
    return 'No Result'

if __name__ == "__main1__":
    calculator('2-6-7*8/2+5')













