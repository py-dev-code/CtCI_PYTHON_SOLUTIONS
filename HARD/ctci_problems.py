from collections import Counter
from random import randint
from bitstring import BitArray

# Problem 17.1: Add Without Plus: Write a function that adds two numbers. You should not use + or any arithmetic
# operators.
'''
    Algorithm:
        1. Add 2 numbers: 759 + 674. If we dont count carry then it will be 323. If we just add carry then it will be 
        1110. If we add 323 and 1110, it will be 1433 which is the result.
        2. This is the base of sum and we need to do this in bits.
        3. How can 2 binary digits be added without carry. Its a ^ b.
        4. How carry gets calculated of 2 binary digits. Its a & b and shifted to left.
        5. This is the recursion base till we get 0 in carry.
'''
def add_without_plus(a, b):
    if b == 0: return a
    sum = a ^ b
    carry = (a & b) << 1
    return add_without_plus(sum, carry)

if __name__ == "__main1__":
    print(add_without_plus(759, 674))


# Problem 17.2: Shuffle: Write a method to shuffle a deck of cards. It must be a perfect shuffle-in other words, each
# of the 52! permutations of the deck has to be equally likely. Assume that you are given a random
# number generator which is perfect.
'''
    Algorithm:
        1. Basis to above point is this: If we have shuffles n-1 cards then to shuffle n cards, all we need to do is 
        take a random card from n-1 cards and swap it with nth card.
        2. In iteration we will do like this: for each element i, swapping array[i] with a random element between 
        0 and i, inclusive.
'''
def shuffle_array(cards):
    for i in range(len(cards)):
        k = randint(0, i)
        cards[k], cards[i] = cards[i], cards[k]

if __name__ == "__main1__":
    arr = [1,2,3,4,5,6,7,8,9,10]
    shuffle_array(arr)
    print(arr)


# Problem 17.3: Random Set: Write a method to randomly generate a set of m integers from an array of size n. Each
# element must have equal probability of being chosen.
'''
    Algorithm:
        Recursive Analysis:
        1. Suppose we have an algorithm that can pull a random set of m elements from an array of size n - 1. 
        How can we use this algorithm to pull a random set of m elements from an array of size n?
        2. We can first pull a random set of size m from the first n - 1 elements. Then, we just need to decide if
        array[n] should be inserted into our subset (which would require pulling out a random element from it).
        An easy way to do this is to pick a random number k from 0 through n. If k < m, then insert array[n] into
        subset[k]. This will both "fairly" (i.e., with proportional probability) insert array[n] into the subset and
        "fairly" remove a random element from the subset.
        
        Iterative Analysis:
        1. In this approach, we initialize an array subset to be the first m elements in original. 
        Then, we iterate through the array, starting at element m, inserting array[i] into the subset at (random) 
        position k whenever k < m.
'''
def random_set(arr, m):
    n = len(arr)
    if m > n: return
    subset = arr[0:m]
    for i in range(m, n):
        k = randint(0, i)
        if k < m:
            subset[k] = arr[i]
    print(subset)

if __name__ == "__main1__":
    arr = [1,2,3,4,5,6,7,8]
    random_set(arr, 4)


# Problem 17.4: Missing Number: An array A contains all the integers from O to n, except for one number which
# is missing. In this problem, we cannot access an entire integer in A with a single operation. The
# elements of A are represented in binary, and the only operation we can use to access them is "fetch
# the jth bit of A[i];"" which takes constant time. Write code to find the missing integer. Can you do it
# in O(n) time?
''' 
    Algorithm:
        1. This algorithm is all about identifying the bit pattern of the numbers in a consisten range.
        2. Take numbers from 0 to 3 with their binaries as: 00, 01, 10, 11. So on each bit position: 
            count(0) = count(1) if n is odd else count(0) = count(1) + 1 if n is even.
        3. Now, we can either remove an odd number or an even number with n being even or odd. So, below 4
        scenarios will appear:
            v even and n even/odd: count(0) <= count(1)
            v odd and n even/odd: count(0) > count(1)
        4. This behavior is true for each bit position.
        5. So, we will take the count of 0s and 1s at each bit position and based on v being even or odd, 
        decide the bit for that bit position as 1/0.
        6. Now, once we decide v is even/odd, we will only search next bit in remaining even/odd numbers.
        7. We will repeat this process till we don't have any numbers left in the list.
        8. First recursion will happen on all the numbers that is n, next one will be on n/2, then on
        n/4 so Time complexity will be O(n).
'''
def find_missing(arr, n):
    def get_bit(arr, i, j):
        b = BitArray(int=arr[i], length=32)
        return int(b[len(b) - j - 1])
    
    def find_missing_util(arr, col):
        if col > 32: return 0
        one_bits = []
        zero_bits = []
        for r in range(len(arr)):
            if get_bit(arr, r, col) == 0:
                zero_bits.append(r)
            else:
                one_bits.append(r)
        if len(zero_bits) <= len(one_bits):
            v = find_missing_util(zero_bits, col + 1)
            return (v << 1) | 0
        else:
            v = find_missing_util(one_bits, col + 1)
            return (v << 1) | 1

    print(find_missing_util(arr, 0))

if __name__ == "__main1__":
    find_missing([0,1,2,3,4,6,7], 7)


# Problem 17.5: Letters and Numbers: Given an array filled with letters and numbers, find the longest subarray with
# an equal number of letters and numbers.
'''
    Algorithm:
        1. Brute Force is to try all subarrays. Time Complexity: O(N^2).
        2. Optimized way: O(N)
            a. At each array index, we need to find out the difference of all letters and all numbers
            came till that point.
            b. If we have this diff for each index then indexes that are farthest with same difference
            are a part of longest subarray.
            c. starting index will start from 1 position after the start index.
            d. We can maintain a Hash Table for each different value of difference with its 1 index.
            So next time, when we see the same difference, we will have oldest index with same difference
            to compare the length of subarrays.
            3. Hash Table will have 1 key in the beginning with key as 0 and value as -1 to make sure
            that difference of 0 is correctly handled.
'''
def letters_and_numbers_subarry_brute_force(arr):
    def check_valid(arr):
        if arr is None or len(arr) == 0: return False
        alpha_count = num_count = 0
        for r in arr:
            if str(r).isalpha(): alpha_count += 1
            if str(r).isnumeric(): num_count += 1
        if alpha_count == num_count:
            return True
        return False
    
    result = []    
    if arr is None or len(arr) == 0: return
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            subarray = arr[i:j]
            if check_valid(subarray) and len(result) < len(subarray):
                result = subarray
    print(result)

def letters_and_numbers_subarray(arr):
    if arr is None or len(arr) == 0: return
    result = []
    hash = {0: -1}
    num_count = alpha_count = 0
    for r in range(len(arr)):
        if str(arr[r]).isnumeric(): num_count += 1
        if str(arr[r]).isalpha(): alpha_count += 1
        diff = num_count - alpha_count
        if diff not in hash:
            hash[diff] = r
        else:
            if len(result) < r - (hash[diff] + 1):
                result = arr[hash[diff] + 1: r + 1]
    print(result)
            
if __name__ == "__main1__":
    letters_and_numbers_subarry_brute_force([1,'A','A',2,'B','C',4,'E','F'])
    letters_and_numbers_subarray(['A','A',2,'B','C',4,'E','F'])
            

# Problem 17.6: Count of 2s: Write a method to count the number of 2s between O and n.
'''
    Algorithm:
        1. Brute Force is Straight Forward for this problem.
        2. To improve this, we need to calculate the presence of 2 at each position.
        3. Ex: 3333
            Position: 2 (starts from 0). First find out the lower and upper round of 10.
            Lower down of 10: 3000 and upper down of 10: 4000
            Find the digit at the position (3).
            a. If digit is less than 2 (say 1) then 2 would come lower down of 10 // 10. 
                100 times from 0 - 1000, 100 times from 1001 - 2000 and 100 times from 2001 - 3000.
            b. If digit is equal to 2 then number of times 2 will come will be 
                result of a + right side number (which is 333 in this case) + 1 i.e. 334 times.
            c. If digit is more than 2 then it will come upper down of 10 // 10. In our case, 
                2's for 3001 - 4000 will also be added which will become 400.
        4. Find out the 2s at each position by same way.
        5. For last position, lower down of 10 should be 0 and upper round will be 10000.
'''
def count_2s_brute_force(n):
    result = 0
    for num in range(2, n + 1):
        result += Counter(str(num))['2']
    print(result)

def count_of_2s(n):
    def count_2s_at_digit(n, digit):        
        power = 10 ** digit
        next_power = 10 * power
        right_number = n % power

        round_down = n - n % next_power
        round_up = round_down + next_power
        digit_num = (n // power) % 10
        if digit_num < 2:
            return round_down // 10
        elif digit_num == 2:
            return round_down // 10 + right_number + 1
        else:
            return round_up // 10

    length = len(str(n))
    result = 0
    for i in range(length):
        num = count_2s_at_digit(n, i)
        result += num
    print(result)

if __name__ == "__main1__":
    count_2s_brute_force(2122)
    count_of_2s(2122)


# Problem 17.7: Baby Names: Each year, the government releases a list of the 10,000 most common baby names
# and their frequencies (the number of babies with that name). The only problem with this is that
# some names have multiple spellings. For example, "John" and ''.Jon" are essentially the same name
# but would be listed separately in the list. Given two lists, one of names/frequencies and the other
# of pairs of equivalent names, write an algorithm to print a new list of the true frequency of each
# name. Note that if John and Jon are synonyms, and Jon and Johnny are synonyms, then John and
# Johnny are synonyms. (It is both transitive and symmetric.) In the final list, any name can be used
# as the "real" name.
# EXAMPLE
# Input:
# Names: John (15), Jon (12), Chris (13), Kris (4), Christopher (19)
# Synonyms: (Jon, John), (John, Johnny), (Chris, Kris), (Chris, Christopher)
# Output: John (27), Kris (36)
'''
    Algorithm:
        1. The trick is to solve this problem with Graph.
        2. For each Synonym pair, create an edge in the graph [either 1 way or 2 way].
        3. Do a complete DFS on the graph as Graph will be disconnected.
        4. For each group of names, dfs will be called once so maintain the result details in the 
        final list.
'''
class Graph(object):
    def __init__(self):
        self.data = {}
    def add_edge(self, node1, node2):
        if node1 not in self.data:
            self.data[node1] = []
        if node2 not in self.data:
            self.data[node2] = []
        self.data[node1].append(node2)
        # self.data[node2].append(node1)
    
def baby_names(name_list, match_list):
    def dfs_util(name_list, data, node, visited, result):
        stack = []
        stack.append(node)
        visited[node] = True
        while len(stack) > 0:
            n = stack.pop()
            value = name_list[n] if n in name_list else 0
            result[len(result) - 1][1] += value
            for nbhr in data[n]:
                if not visited[nbhr]:
                    stack.append(nbhr)
                    visited[nbhr] = True
    
    graph = Graph()
    for node1, node2 in match_list:
        graph.add_edge(node1, node2)    
    visited = {}
    for node in graph.data:
        visited[node] = False
    result = []
    for node in graph.data:
        if not visited[node]:
            result.append([node, 0])
            dfs_util(name_list, graph.data, node, visited, result)

    print(result)    

if __name__ == "__main1__":
    name_list = {'John': 15, 'Jon': 12, 'Chris': 13, 'Kris': 4, 'Christopher': 19}
    match_list = [('Jon', 'John'), ('John', 'Johnny'), ('Chris', 'Kris'), ('Chris', 'Christopher')]
    baby_names(name_list, match_list)


# Problem 17.8: Circus Tower: A circus is designing a tower routine consisting of people standing atop one another's
# shoulders. For practical and aesthetic reasons, each person must be both shorter and lighter than
# the person below him or her. Given the heights and weights of each person in the circus, write a
# method to compute the largest possible number of people in such a tower.
class Person(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w
    def __repr__(self):
        return f'{self.h}H:{self.w}W'
    def __lt__(self, other):
        return self.h < other.h and self.w < other.w
    def __eq__(self, other):
        return self.h == other.h and self.w == other.w

def circus_tower(persons):
    def can_adjust(down, up, persons):
        return persons[up] < persons[down]
    
    def get_tower_height(persons, index, len_map, result):
        if len_map[index] is not None:
            return len_map[index]
        length = 1
        result.append(persons[index])
        for i in range(index+1, len(persons)):
            if can_adjust(index, i, persons):
                length += get_tower_height(persons, i, len_map, result)
                break
        len_map[index] = length
        return length

    persons.sort(reverse=True)
    max_length = 0
    len_map = {}
    for r in range(len(persons)):
        len_map[r] = None
    
    result = []
    results = [None]
    for r in range(len(persons)):
        length = get_tower_height(persons, r, len_map, result)
        if length > max_length:
            max_length = length
            results[0] = result
            result = []
    
    print(f'Max Length is : {max_length}')
    print(f'Person of the Longest Tower are: {results[0]}')

if __name__ == "__main1__":
    persons = [Person(1,2), Person(3,4), Person(0,0), Person(0,1), Person(1,2)]
    circus_tower(persons)        


# Problem 17.9: Kth Multiple: Design an algorithm to find the kth number such that the only prime factors are 3, 5,
# and 7. Note that 3, 5, and 7 do not have to be factors, but it should not have any other prime factors.
# For example, the first several multiples would be (in order) 1, 3, 5, 7, 9, 15, 21.
'''
    Algorithm:
        1. This algorithm is really tricky but once we get the pattern, implementation is very straight forward.
        2. Problem is to build a series that can not have any prime number other than 3,5 and 7 as its factor.
        So, number will follow the pattern 3^a + 5^b + 7^c.
        3. Brute Force is to build all possible numbers upto 3^k + 5^k + 7^k where k in (0,k) and sort them and 
        get the kth element. Complexity will be O(k^3).
        4. Now lets find the pattern:
            3^0 + 5^0 + 7^0                 1
            3^1 + 5^0 + 7^0     3*(3^0)     3
            3^0 + 5^1 + 7^0     5*(5^0)     5
            3^0 + 5^0 + 7^1     7*(7^0)     7
            3^2 + 5^0 + 7^0     3*(3^1)     9
            3^1 + 5^1 + 7^0     3*(5^1)     15
        So, each number is formed after multiplying a previous number in the list to 3/5/7. 
        That is kth number = {1st, 2nd.... (k-1)th number} * 3/5/7
        5. Now, to implement this finding when we find an element then we can multiply it via 3,5 and 7 and
        store those numbers in a separate list and next number will be the minimum of that list.
        6. This approach will be better than Brute Force.
        7. Next improvement can be to find the minimum element.
        8. If we maintain 3 list: q3, q5 and q7 to store multiplies of each separately then we just need
        to compare their 1st element to get the result as all other elements will be bigger.
        9. In this case, we will have some duplicates as (num * 5) * 3 will also come in (num * 3) * 5.
        10. To avoid duplicates, we can do following:
            a. whenever we take num from q3 then insert num*3 in q3, num*5 in q5 and num*7 in q7
            as they will be occurring 1st time.
            b. whenever we take num from q5 then insert num*5 in q5 and num*7 in q7
            as num*3 have already gone in q3.
            c. Similarly, for a num of q7, only add num*7 in q7.
        11. For each index, we are doing constant steps so Time Complexity should be O(1).
'''
def kth_multiple(k):
    def get_min(q3, q5, q7):
        if q3[0] < q5[0] and q3[0] < q7[0]:
            value = q3[0]
            q3.remove(q3[0])
            q3.append(value * 3)
            q5.append(value * 5)
            q7.append(value * 7)
        elif q5[0] < q3[0] and q5[0] < q7[0]:
            value = q5[0]
            q5.remove(q5[0])
            q5.append(value * 5)
            q7.append(value * 7)
        else:
            value = q7[0]
            q7.remove(q7[0])
            q7.append(value * 7)
        return value
    
    arr = [1]
    q3 = [3]
    q5 = [5]
    q7 = [7]

    i = 0
    while i < k:
        arr.append(get_min(q3, q5, q7))
        i += 1
    print(arr)

if __name__ == "__main1__":
    kth_multiple(100)


# Problem 17.10: Majority Element: A majority element is an element that makes up more than half of the items in
# an array. Given a positive integers array, find the majority element. If there is no majority element,
# return -1. Do this in O(N) time and 0(1) space.
# Input: 1 2 5 9 5 9 5 5 5
# Output: 5
'''
    Algorithm:
        1. One of trickiest problem.
        2. Since we need O(N) time and O(1) space so no point of using any other storage.
        3. We will assume that Each element is a majority element of a subarray and it is at the beginning 
        of that subarray.
        4. With this assumption, we will traverse the array and as soon as we find that its not a majority
        element, we will move to the next element.
        5. Here, one more observation to notice:
            Ex array: 3,1,7,1,1,7,7,3,7,7,7
            a. We start with 3 assuming that 3 is majority element and its the starting of the subarray.
            b. As soon as we reach index 2, we know that 3 is no more a majority so we discard 3 and move
            to validate 1.
            c. Important thing is since 3 is not a majority so 1 also can not be majority as we are 
            discarding the number as soon as 3 is appearing less than half elements so this eliminates
            both 3 and 1.
            d. Next step is to start validate 7.
            e. We continue like this and in the end we will find a majority element which may or may not
            be the majority element. But no other element can be majority in the array for sure.
            f. Scan array one more time to check if result is really majority or not.
'''
def find_majority_element(arr):
    def get_possible_num(arr):
        majority = 0
        count = 0
        for r in arr:
            if count == 0:
                majority = r
            if r == majority:
                count += 1
            else:
                count -= 1
        return majority

    def validate(arr, num):
        count = 0
        for r in arr:
            if r == num: count += 1
        return count > len(arr) // 2
    
    num = get_possible_num(arr)
    if validate(arr, num):
        return num
    return -1

if __name__ == "__main1__":
    print(find_majority_element([1,5,5,1,1,5,1,1]))


# Problem 17.11: Word Distance: You have a large text file containing words. Given any two words, find the shortest
# distance (in terms of number of words) between them in the file. If the operation will be repeated
# many times for the same file (but different pairs of words), can you optimize your solution?
'''
    Algorithm:
        1. Create a Hash Table with keys as all the words and their appeance index will be a list in 
        sorted order.
        2. To search min distance between 2 words, we need to implement an algorithm to find an 
        element pair with min difference in 2 sorted arrays.
        3. This can be implemented via moving 2 pointers approach. Time Complexity will be O(A+B).
'''
def word_distance(book, word1, word2):
    def create_hash_table(book):
        result = {}
        for idx, word in enumerate(book.split(' ')):
            if word not in result:
                result[word] = [idx]
            else:
                result[word].append(idx)
        return result

    def get_min_distance(arr1, arr2, pair):
        distance = arr1[0] - arr2[0]
        pair[0] = [arr1[0], arr2[0]]
        i = j = 0
        while i < len(arr1) and j < len(arr2):
            diff = arr1[i] - arr2[j]
            if abs(diff) < abs(distance):
                distance = diff
                pair[0] = [arr1[i], arr2[j]]
            if diff > 0: 
                j += 1
            else: 
                i += 1            
        return distance

    ht = create_hash_table(book)
    if word1 not in ht or word2 not in ht: print('-1')
    pair = [[]]
    d = get_min_distance(ht[word1], ht[word2], pair)
    print(f'Min Distance: {abs(d)} with indexes: {pair[0]}')

if __name__ == "__main1__":
    word_distance('a c e d e g k l m a b d', 'a', 'd')


# Problem 17.12: BiNode: Consider a simple data structure called BiNode, which has pointers to two other nodes. The
# data structure BiNode could be used to represent both a binary tree (where node1 is the left node
# and node2 is the right node) or a doubly linked list (where node1 is the previous node and node2
# is the next node). Implement a method to convert a binary search tree (implemented with BiNode)
# into a doubly linked list. The values should be kept in order and the operation should be performed
# in place (that is, on the original data structure).
'''
    Algorithm:
        1. A classic recursion problem.
        2. Mainly, at each node we need to adjust its node1 and node2 pointers. node1 pointer is right most
        node in left subtree and node2 is left most node in right subtree. Repeat this for all the roots
        in the recursion and our doubly linked list will be ready.
        3. Now, since we need to return the head of the list and our recursion method can only return the head 
        so to save some time, we can just save the head pointer before starting the pointer adjustments. This 
        will take O(logN) time in worst case (if all tree nodes are in left side).
        4. Conversion will roughly take logN time for a leaf node and we are anyways touching all the nodes so
        time complexity should be O(NlogN).
        5. Complexity of O(NlogN) can be acheived only when Tree is balanced else it can go up to O(N^2) in worst 
        case.
        6. An O(N) algorithm exist for the coversion.
'''
class BiNode(object):
    def __init__(self, data, node1=None, node2=None):
        self.data = data
        self.node1 = node1
        self.node2 = node2
    def __repr__(self):
        return str(self.data)
    
def chande_bst_to_doubly_ll(root):
    def get_head(root):
        head = root
        while head.node1:
            head = head.node1
        return head

    def convert_to_list(root):
        if root is None: return
        convert_to_list(root.node1)
        convert_to_list(root.node2)        

        prev = root.node1
        if prev:
            while prev.node2:
                prev = prev.node2
            prev.node2 = root
            root.node1 = prev

        next = root.node2
        if next:
            while next.node1:
                next = next.node1
            next.node1 = root
            root.node2 = next   

    head = get_head(root)
    convert_to_list(root)    
    result = []
    while head:
        result.append(str(head.data))
        head = head.node2
    print(' <-> '.join(result))

if __name__ == "__main1__":
    root = BiNode(4)
    root.node1 = BiNode(2)
    root.node1.node1 = BiNode(1)
    root.node1.node2 = BiNode(3)
    root.node2 = BiNode(6)
    root.node2.node1 = BiNode(5)
    root.node2.node2 = BiNode(7)
    chande_bst_to_doubly_ll(root)


# Problem 17.13: Re-Space: Oh, no! You have accidentally removed all spaces, punctuation, and capitalization in a
# lengthy document. A sentence like "I reset the computer. It still didn't boot!"
# became "iresetthecomputeritstilldidntboot''. You'll deal with the punctuation and capitalization
# later; right now you need to re-insert the spaces. Most of the words are in a dictionary but
# a few are not. Given a dictionary (a list of strings) and the document (a string), design an algorithm
# to unconcatenate the document in a way that minimizes the number of unrecognized characters.
# EXAMPLE
# Input jesslookedjustliketimherbrother
# Output: jess looked just like tim her brother (7 unrecognized characters)
'''
    Algorithm:
        1. Slightly complicated problem to implement.
        2. Think like doc has just 1 word with 2 letters and its not in the list then output will be the whole word 
        without space as we need to minimize the invalid word counts so we add space only when we have found
        a valid word.
        3. We can build output for index i by checking the output from index i+1 till end.
        4. If word till index i is valid then we will add a space else it will just be appeneded without space.
        5. We will maintaint best invalid count and best_parsed values for each index so as to replace the
        values whenever we find a better option.
        6. Time Complexity: O(N^2)
'''
class ParseResult(object):
    def __init__(self, invalid, parsed):
        self.invalid = invalid
        self.parsed = parsed
    
def best_split(word_list, doc):
    def split(word_list, doc, start, hash_list):
        if start >= len(doc):
            return ParseResult(0, '')
        if hash_list[start] is not None:
            return hash_list[start]

        best_invalid = None
        best_parsing = ''
        partial = ''
        index = start
        while index < len(doc):
            c = doc[index]
            partial += c
            invalid = 0 if partial in word_list else len(partial)            
            if best_invalid is None or invalid <= best_invalid:
                result = split(word_list, doc, index + 1, hash_list)
                if best_invalid is None or invalid + result.invalid <= best_invalid:
                    best_invalid = invalid + result.invalid
                    best_parsing = partial + ' ' + result.parsed
                    if best_invalid == 0:
                        break
            index += 1
        hash_list[start] = ParseResult(best_invalid, best_parsing)
        return hash_list[start]
    
    hash_list = [None for x in range(len(doc))]
    result = split(word_list, doc, 0, hash_list)
    string =  '' if result is None else result.parsed
    print(string)

if __name__ == "__main1__":
    best_split(['looked', 'just', 'like', 'her', 'brother'], 'jesslookedjustliketimherbrother')
    best_split(['am'], 'abc')


# Problem 17.14: Smallest K: Design an algorithm to find the smallest K numbers in an array.
'''
    Algorithm:
        There are various approaches to solve this problem.
        1. Heap:
            a. Create a min heap from the array in O(n) time.
            b. Remove the root of heap for k time so last k element of the array will be smallest k elements.
        
        2. Ranking for Unique:
            a. Find the element with kth rank in the array in O(n) time.
                i. Find a pivot in the array.
                ii. Partition all smaller elements to left of the pivot.
                iii. If left size is equal to rank then return max element from the left.
                iv. If left size is more than rank then find rank between left and left_end.
                v. If left size is less than rank then (rank - left_size) rank in left_end+1 to right part.
                vi. This approach doesnt work in duplicate array.
                vii. Same approach needs to be applied for Quicksort.
            b. Scan the array and add all elements less than or equal to kth rank element in the result.

        3. Ranking for Duplicate values:
            Exactly same as number 3 but partition will break the array in 3 parts: left, middle and right.
            left will be smaller than pivot, middle is same and right is more than pivot.
            Slight changes in rank function to handle the 3 partitions.
'''
def smallest_k_ranking(arr, k):
    def partition(arr, left, right, pivot):
        while left <= right:
            if arr[left] > pivot:
                arr[left], arr[right] = arr[right], arr[left]
                right -= 1
            elif arr[right] <= pivot:
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
            else:
                left += 1
                right -= 1
        return left - 1
    
    def rank_util(arr, left, right, rank):
        pivot = arr[randint(left, right)]
        left_end = partition(arr, left, right, pivot)
        left_size = left_end - left + 1
        if rank == left_size:
            return max(arr[left: left_end + 1])
        elif rank < left_size:
            return rank_util(arr, left, left_end, rank)
        else:
            return rank_util(arr, left_end + 1, right, rank - left_size)
    
    if k < 1 or k > len(arr):
        return 
    threshold = rank_util(arr, 0, len(arr) - 1, k)
    result = []
    count = 0
    for r in arr:
        if r <= threshold:
            if count == k:
                break
            result.append(r)
            count += 1
    print(result)

def smallest_k_ranking_duplicate(arr, k):
    class PartitionResult(object):
        def __init__(self, left_size, middle_size):
            self.left_size = left_size
            self.middle_size = middle_size

    def partition(arr, start, end, pivot):
        left = start
        middle = start
        right = end
        while middle <= right:
            if arr[middle] < pivot:
                arr[left], arr[middle] = arr[middle], arr[left]
                middle += 1
                left += 1
            elif arr[middle] > pivot:
                arr[middle], arr[right] = arr[right], arr[middle]
                right -= 1
            elif arr[middle] == pivot:
                middle += 1
        return PartitionResult(left - start, right - left + 1)
    
    def rank_util(arr, left, right, rank):
        pivot = arr[randint(left, right)]
        partition_result = partition(arr, left, right, pivot)
        left_size = partition_result.left_size
        middle_size = partition_result.middle_size

        if rank < left_size:
            return rank_util(arr, left, left + left_size - 1, rank)
        elif rank < left_size + middle_size:
            return pivot
        else:
            return rank_util(arr, left + left_size + middle_size, right, rank - left_size - middle_size)
    
    if k < 1 or k > len(arr):
        return 
    threshold = rank_util(arr, 0, len(arr) - 1, k)
    result = []
    count = 0
    for r in arr:
        if r <= threshold:
            if count == k:
                break
            result.append(r)
            count += 1
    print(result)        

if __name__ == "__main1__":
    smallest_k_ranking([10,8,7,6,9,5,4,2,3,1], 5)
    smallest_k_ranking_duplicate([1,2,2,2,2,2,2,2,2,2,2,2], 5)


# Problem 17.15: Longest Word: Given a list of words, write a program to find the longest word made of other words
# in the list.
'''
    Algorithm:
        1. Difficult to build the approach but easier to implement.
        2. First of all we will apply the checks from longest word so we will sort the list by reverse order.
        3. For a faster lookup, we will put the word list in a hash table.
        4. Now to check if a word is composite or not:
            a. we will try all word combinations first of all.
            b. 1st combination in word "amount" will be "a" and "mount".
            c. now if a is not in the hash then no need to check further. Say, a is in the list then 
            we cant return False if mount is not in the list because possibly mo and unt can be in the list.
            So, just apply the check method recursively.
            d. One point to note it that, recursive method should return True if it finds the word in the 
            list but not when we are checking for the original word so track original word with a flag.
            e. Slight improvements can be made when we use memoization by putting word in the hash that is
            not composite with status as False.
        5. Both recursive methods memo and without memo are implemented below.
'''
def get_longest_composite_word(word_list):
    def is_composite(word, word_hash, original_word):
        if word in word_hash and not original_word:
            return word_hash[word]
        for i in range(len(word)):
            left = word[0:i+1]
            right = word[i+1:len(word)]
            if left in word_hash and word_hash[left] and is_composite(right, word_hash, False):
                return True
        if word not in word_hash:
            word_hash[word] = False
        return False

    def is_composite_non_memo(word, word_hash, original_word):
        if word in word_hash and not original_word:
            return True            
        for i in range(len(word)):
            left = word[0:i+1]
            right = word[i+1:len(word)]
            if left in word_hash and is_composite(right, word_hash, False):
                return True
        return False

    word_hash = {}
    for word in word_list:
        word_hash[word] = True
    word_list.sort(key=lambda x: len(x), reverse=True)
    result = ''
    for word in word_list:
        if is_composite(word, word_hash, True):
        # if is_composite_non_memo(word, word_hash, True):
            result = word
            print(result)
            return
    print("No word found!")

if __name__ == "__main1__":
    get_longest_composite_word(['a', 'am', 'amorph', 'acycli', 'cyclics', 'amplemounting', 'ample', 'mount', 'ing'])

            
# Problem 17.16: The Masseuse: A popular masseuse receives a sequence of back-to-back appointment requests
# and is debating which ones to accept. She needs a 15-minute break between appointments and
# therefore she cannot accept any adjacent requests. Given a sequence of back-to-back appointment
# requests (all multiples of 15 minutes, none overlap, and none can be moved), find the optimal
# (highest total booked minutes) set the masseuse can honor. Return the number of minutes.
# EXAMPLE
# Input: {30, 15, 60, 75, 45, 15, 15, 45}
# Output: 180 minutes ({30, 60, 45, 45}).  
''' 
    Algorithm:
        1. Problem is difficult to understand but quite easy to solve.
        2. Thing with the appointments is if you pick current appointment then you cant pick the 
        next one due to 15 min break.
        3. With this condition, we need to find the max of all given appointment times.
'''
def max_masseuse_minute(appointments):
    def minutes_util(appointments, index, hash):
        if index >= len(appointments):
            return 0
        if hash[index] is not None:
            return hash[index]

        best_with = appointments[index] + minutes_util(appointments, index + 2, hash)
        best_without = minutes_util(appointments, index + 1, hash)
        hash[index] = max(best_with, best_without)
        return hash[index]        

    hash = [None for x in appointments]
    result = minutes_util(appointments, 0, hash)
    print(result)

if __name__ == "__main1__":
    appointments = [30, 15, 60, 75, 45, 15, 15, 45]
    max_masseuse_minute(appointments)


# Problem 17.17: Multi Search: Given a string b and an array of smaller strings T, design a method to search b for
# each small string in T.
'''
    Algorithm:
        A. Brute Force:
            1. Search each small word in big string starting from each location.
            2. Create a Dictionary for each small word with value is a list of all index positions in big string
            to return the output.
            3. Time: O(kbt) where k: number of small words, b: length of big string and t is length of biggest small
            word.
        
        B. Trie Approach:
            1. Create a Trie with all the small words.
            2. Starting from each index in big string, apply a search in Trie.
            3. If index letter is a word then add it in result. If index letter is also a prefix
            then move this word for next iteration else clear the word.
            4. In this, we will search 1 word multiple times so maintain a set in dictionary as values.
            5. Time in making Trie: O(kt) and time in searching all big string: O(bt)
                So complete time: O(kt + bk) which is better than O(kbt).
'''
def multi_search_brute_force(big, smalls):
    def search(big, small):
        locations = []
        start = 0
        while start < len(big):
            end = start + len(small)
            if small == big[start: end]:
                locations.append(start)
            start += 1
        return locations
    
    result = {}
    for small in smalls:
        index_list = search(big, small)
        result[small] = index_list
    for item in result.items():
        print(item)

class TrieNode(object):
    def __init__(self):
        self.children = [None] * 26
        self.isLeaf = False

class Trie(object):
    def __init__(self):
        self.root = TrieNode()    
    def insert(self, word):
        def insert_util(root, word):
            for l in word:
                index = ord(l) - ord('a')
                if root.children[index] is None:
                    root.children[index] = TrieNode()
                root = root.children[index]
            root.isLeaf = True
        insert_util(self.root, word)
    def search(self, word):
        def search_util(root, word):
            for l in word:
                index = ord(l) - ord('a')
                if root.children[index] is None:
                    return False
                root = root.children[index]
            return root.isLeaf
        return search_util(self.root, word)
    def starts_with(self, prefix):
        def starts_with(root, prefix):
            for l in prefix:
                index = ord(l) - ord('a')
                if root.children[index] is None:
                    return False
                root = root.children[index]
            if root is None: return False
            return True
        return starts_with(self.root, prefix)

def multi_search_trie(big, smalls):
    def find_strings_from_location(big, index, trie, result):
        word = ''
        for i in range(index, len(big)):
            word += big[i]
            if trie.search(word):
                if word in result:
                    result[word].add(i - len(word) + 1)
                else:
                    result[word] = set([i - len(word) + 1])
            if not trie.starts_with(word):
                word = ''

    trie = Trie()
    for small in smalls:
        trie.insert(small)
    result = {}
    word = ''
    for i in range(len(big)):
        find_strings_from_location(big, i, trie, result)
    for item in result.items():
        print(item)

if __name__ == "__main1__":
    multi_search_brute_force('mississippi', ["is", "ppi", "hi", "sis", "i", "ssippi"])
    print("===")
    multi_search_trie('mississippi', ["is", "ppi", "hi", "sis", "i", "ssippi"])


# Problem 17.18: Shortest Supersequence: You are given two arrays, one shorter (with all distinct elements) and one
# longer. Find the shortest subarray in the longer array that contains all the elements in the shorter
# array. The items can appear in any order.
# EXAMPLE
# Input: {1, 5, 9} and {7, 5, 9, 0, 2, 1, 3, "5, 7, 9, 1", 1, 5, 8, 8, 9, 7}
# Output:[7, 10] (range of index in bigger array)
'''
    Algorithm: 
        1. This problem can be solved by some pre computing.
        2. For each element in small array, we will create a list with same length as big array. This 
        list value at a given index will tell the index where the small element can be found next. To do this,
        we need to scan the array in reverse way and list will be ready in O(N) time.
        3. Once this list is ready for all the small elements, we need to run a loop for length of big
        array, for each index, we will take the max from all list's same index elements and subtract current
        index from this max. This tells us the shortest supersequence length starting from the current index.
        4. Repeat it until the completed array length (exit if None is there in one of the lists) and keep
        updating the shortest supersequence length and range of indexes.
'''
def shortest_supersequence(big, small):
    closure_data = []
    for i in small:
        closure_data.append([None for x in big])
    for i in range(len(small)):
        curr = None
        for j in reversed(range(len(big))):
            if big[j] == small[i]:
                curr = j
            closure_data[i][j] = curr
    result_range = []
    result_length = None
    for j in range(len(big)):
        try: 
            compare_list = []
            for i in range(len(closure_data)):
                compare_list.append(closure_data[i][j])
            min_closure = max(compare_list) - j
            if result_length is None or min_closure < result_length:
                result_length = min_closure
                result_range = [j, j + min_closure]
        except: 
            break
    print(f'Shortest Supersequence is: {big[result_range[0]: result_range[1]+1]} with length: {result_length + 1}')

if __name__ == "__main1__":
    shortest_supersequence([7,5,9,0,2,1,3,5,7,9,1,1,5,8,8,9,7], [1,5,9])


# Problem 17.19: Missing Two: You are given an array with all the numbers from 1 to N appearing exactly once,
# except for one number that is missing. How can you find the missing number in O(N) time and
# 0(1) space? What if there were two numbers missing?
'''
    Algorithm:
        A. 1 Missing:   
            1. Take the sum of the array and sutract it from sum of 1st N integers and we will get the missing 
            number.
        B. 2 Missing:
            1. Do the same step and difference will give use a + b value.
            2. Repeat the same for squares of numbers and difference will give use a^2 + b^2.
            3. Now it will be a quadratic equation. We can solve it in O(1) time by using formula:
                ax^2 + bx + c = 0
                x value will be: 
                    (-b +- (b^2 - 4ac)^(1/2)/2a
                read as "minus b +/- under root of b square minus 4ac" divided by 2a.
            4. This gives 2 values and that will be of a and b. We will use only + sign to get a and
            use a+b equation to get b.
            5. Equation solution is not coming correctly right now so need to fix that part.
'''
# def find_2_missing(arr, N):
#     add = (N*(N+1)) / 2
#     array_add = sum(arr)
#     normal_sum = add - array_add

#     square_add = 0
#     for r in range(1,N+1):
#         square_add += r*r
#     array_square_add = sum([x*x for x in arr])
#     square_sum = square_add - array_square_add
#     # Quadratic Equation is:
#     #     2x^2 -2sum1 x + sum1^2 - sum2 = 0
#     a = 2
#     b = -2 * normal_sum
#     c = (normal_sum * normal_sum) - square_sum

#     x = (-b + (b**2 - 4*a*c)**(0.5) ) / 2*a
#     y = normal_sum - x
#     print(f'x: {x} and y: {y}')

if __name__ == "__main1__":
    find_2_missing([1,4], 4)


# Problem 17.20: Continuous Median: Numbers are randomly generated and passed to a method. Write a program
# to find and maintain the median value as new values are generated.
'''
    Algorithm:
        1. This problem can be easily solved by maintaining 2 Heaps in our class. 
        2. One is Max Heap that will save all elements less than Median and one is Min Heap that will
        save all elements more than median.
        3. Median will always be updated on inserting a new element.
        4. Median will be average of root of both the heaps if their size is same else it will be root 
        of max heap. To make it work, we need to maintain the size of both the heaps so that max heap can 
        be 1 size more or equal in size of min heap and min heap will always be smaller or equal to max
        heap in size.
        5. Implementation is lengthy due to entire min heap and max heap implementation but flow is 
        very straight forward.
'''
class Median(object):
    def __init__(self):
        self.median = None
        self.min_heap = []
        self.min_size = 0
        self.max_heap = []
        self.max_size = 0
    
    def insert(self, value):
        if self.max_size == 0:
            self.max_heap.append(value)
            self.median = value
            self.max_size += 1
        else:
            if value <= self.median:
                self.insert_max_heap(value)                
            else:
                self.insert_min_heap(value)
        self.adjust_heaps()
        self.update_median()       
    
    def insert_min_heap(self, value):
        def swim_min_heap(i):
            if i <= 0: return
            p = (i - 1) // 2
            if self.min_heap[p] > self.min_heap[i]:
                self.min_heap[p], self.min_heap[i] = self.min_heap[i], self.min_heap[p]
                swim_min_heap(p)
        if len(self.min_heap) == 0:
            self.min_heap.append(value)            
            self.min_size += 1
        else:
            self.min_heap.append(value)    
            self.min_size += 1        
            swim_min_heap(self.min_size - 1)       

    def insert_max_heap(self, value):
        def swim_max_heap(i):
            if i <= 0: return
            p = (i - 1) // 2
            if self.max_heap[p] < self.max_heap[i]:
                self.max_heap[p], self.max_heap[i] = self.max_heap[i], self.max_heap[p]
                swim_max_heap(p)
        self.max_heap.append(value)
        self.max_size += 1
        swim_max_heap(self.max_size - 1)
    
    def remove_from_min_heap(self):
        def sink_min_heap(i):
            left = i*2 + 1
            right = i*2 + 2
            if left >= self.min_size: return
            if right >= self.min_size or self.min_heap[left] <= self.min_heap[right]:
                child = left
            else:
                child = right
            if self.min_heap[child] <= self.min_heap[i]:
                self.min_heap[child], self.min_heap[i] = self.min_heap[i], self.min_heap[child]
                sink_min_heap(child)

        value = self.min_heap[0]
        self.min_heap[0], self.min_heap[self.min_size - 1] = self.min_heap[self.min_size - 1], self.min_heap[0]
        self.min_heap[self.min_size - 1] = None # Else Duplicates will cause issues
        self.min_heap.remove(self.min_heap[self.min_size - 1])
        self.min_size -= 1
        sink_min_heap(0)
        return value

    def remove_from_max_heap(self):
        def sink_max_heap(i):
            left = i*2 + 1
            right = i*2 + 2
            if left >= self.max_size: return
            if right >= self.max_size or self.max_heap[left] >= self.max_heap[right]:
                child = left
            else:
                child = right
            if self.max_heap[child] >= self.max_heap[i]:
                self.max_heap[child], self.max_heap[i] = self.max_heap[i], self.max_heap[child]
                sink_max_heap(child)

        value = self.max_heap[0]
        self.max_heap[0], self.max_heap[self.max_size - 1] = self.max_heap[self.max_size - 1], self.max_heap[0]       
        self.max_heap[self.max_size - 1] = None # Else Duplicates will cause issues
        self.max_heap.remove(self.max_heap[self.max_size - 1])
        self.max_size -= 1      
        sink_max_heap(0)
        return value        

    def adjust_heaps(self):
        if self.min_size > self.max_size:
            value = self.remove_from_min_heap()
            self.insert_max_heap(value)
        elif self.max_size > self.min_size + 1:
            value = self.remove_from_max_heap()
            self.insert_min_heap(value)

    def update_median(self):
        if self.min_size == self.max_size:
            self.median = (self.min_heap[0] + self.max_heap[0]) / 2
        else:
            self.median = self.max_heap[0]

if __name__ == "__main1__":
    m = Median()
    arr = [2,3,4,1,5,6,7,4,2]
    for r in range(len(arr)):
        m.insert(arr[r])
        print(f'array is: {sorted(arr[0:r+1])} and Median is: {m.median}')
    

# Problem 17.21: Volume of Histogram: Imagine a histogram (bar graph). Design an algorithm to compute the
# volume of water it could hold if someone poured water across the top. You can assume that each
# histogram bar has width 1.
# EXAMPLE
# lnput: {0, 0, 4, 0, 0, 6, 0, 0, 3, 0, 5, 0, 1, 0, 0, 0}
# Output: 26
'''
    Algorithm:
        1. Very nice algorithm to implement.
        2. If we can figure out the amount of water above each histogram bar then we can get the volume.
        3. Water above any bar is determined by 2 values: left tallest tower and right tallest tower. If
        we know these 2 values then minimum of these will determine the water that will come on this bar.
        Simply take the min and subtract the index value from it to get the final amount of water above
        any bar.
        4. To perform step 3, we need to know left tallest bar and right tallest bar at all indexes in advace.
        5. Well, scan the array once to create a storage array that will keep the tallest height from left
        at a given index.
        6. Scan the array from right and perform step 3. We can keep track of right tallest by a simple variable
        so no need to store the tallest from right heights in a storage array.
        7. After the second array scan is done, we will get our volume.
'''
def histogram_volume(arr):
    max_left = arr[0]
    left_heights = []
    for r in range(len(arr)):
        max_left = max(max_left, arr[r])
        left_heights.append(max_left)
    
    max_right = arr[len(arr) - 1]
    volume = 0
    for r in reversed(range(len(arr))):
        max_right = max(max_right, arr[r])
        volume += min(max_right, left_heights[r]) - arr[r]
    print(volume)

if __name__ == "__main1__":
    arr = [0, 0, 4, 0, 0, 6, 0, 0, 3, 0, 5, 0, 1, 0, 0, 0]
    histogram_volume(arr)


# Problem 17.22: Word Transformer: Given two words of equal length that are in a dictionary, write a method to
# transform one word into another word by changing only one letter at a time. The new word you get
# in each step must be in the dictionary.
# EXAMPLE
# Input: DAMP, LIKE
# Output: DAMP-> LAMP-> LIMP-> LIME-> LIKE
'''
    Algorithm:
        There is no shortcut for this search so we have to go via DFS/BFS.

        1. DFS:
            a. DFS will be straight forward only thing is we will only search if next word is one edit 
            away from the current word and it will be done.
        
        2. BFS: 
            a. To apply BFS in this case, we need to move the valid one edit away words from the given 
            word so for that we need to transform our word dictionary a bit that is for each word, we 
            will add a list of valid one edit away birds.
            b. Our BFS will use this list to complete the search.

        3. Bidirectional BFS:
            a. If distance between source and dest is 4 and a total of 15 possible words are there in 
            our dict then normal BFS will take a max of 15^4 searches.
            b. If we do a bidirectional BFS then it will only take 15^2 + 15^2 searches so it will be 
            the most optimized algorithm in this case.
'''
def word_transformer_dfs(word_list, word1, word2):
    def is_one_edit_away(word1, word2):
        diff = False
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                if diff: return False
                diff = True
        return True
    
    def dfs_util(word_dict, word1, word2, cache, path):
        path.append(word1)
        if word1 == word2:
            return True
        for word in word_dict:
            if is_one_edit_away(word, word1) and word not in cache:
                cache[word] = True
                result = dfs_util(word_dict, word, word2, cache, path)
                if result: 
                    return True
        path.pop()
        del cache[word1]
        return False

    word_dict = {}
    for word in word_list:
        word_dict[word] = True
    cache = {}
    path = []
    cache[word1] = True
    if dfs_util(word_dict, word1, word2, cache, path):
        print(' -> '.join(path))
    else:
        print('No Transformation is possible!')

def word_transformer_bfs(word_list, word1, word2):
    def transform_word_dict(word_dict):
        for word in word_dict:
            for i in range(len(word)):
                for key in word_dict:
                    if len(key) == len(word) and key != word:
                        string = word[0:i] + key[i] + word[i+1: len(word)]
                        if string != word and string not in word_dict[word]:
                            word_dict[word].append(string)

    def bfs_util(word_dict, word1, word2, cache, path):
        if word1 not in word_dict: return False
        path.append(word1)
        if word1 == word2:
            return True
        for word in word_dict[word1]:
            if word not in cache:
                cache[word] = True
                result = bfs_util(word_dict, word, word2, cache, path)
                if result:
                    return True
        path.pop()
        del cache[word1]
        return False
    
    word_dict = {}
    for word in word_list:
        word_dict[word] = []
    transform_word_dict(word_dict)
    cache = {}
    path = []
    cache[word1] = True
    if bfs_util(word_dict, word1, word2, cache, path):
        print(' -> '.join(path))
    else:
        print('No Transformation is possible!')

def word_transformer_bi_directional_bfs(word_list, word1, word2):
    class PathNode(object):
        def __init__(self, word=None, prev=None):
            self.word = word
            self.prev = prev
    
    class BFSData(object):        
        def __init__(self):            
            self.to_visit = []
            self.visited = {}
        def add_node(self, word):
            node = PathNode(word)
            self.to_visit.append(node)
            self.visited[word] = node
        def is_finished(self):
            return len(self.to_visit) == 0

    def transform_word_dict(word_dict):
        for word in word_dict:
            for i in range(len(word)):
                for key in word_dict:
                    if len(key) == len(word) and key != word:
                        string = word[0:i] + key[i] + word[i+1: len(word)]
                        if string != word and string not in word_dict[word]:
                            word_dict[word].append(string)

    def merge_paths(source, dest, common):
        end1 = source.visited[common]
        end2 = dest.visited[common]
        result = []
        while end1:
            result.append(end1.word)
            end1 = end1.prev
        result = list(reversed(result))
        result.pop()
        while end2:
            result.append(end2.word)
            end2 = end2.prev
        return result

    def search_level(word_dict, first, second):
        count = len(first.to_visit)
        for i in range(count):
            node = first.to_visit.pop(0)
            word = node.word            
            if word in second.visited:
                return word
            if word in word_dict:
                for nbhr in word_dict[word]:
                    if nbhr not in first.visited:
                        next = PathNode(nbhr, node)
                        first.to_visit.append(next)
                        first.visited[nbhr] = next                            

    word_dict = {}
    for word in word_list:
        word_dict[word] = []
    transform_word_dict(word_dict)

    source_data = BFSData()
    source_data.add_node(word1)
    dest_data = BFSData()
    dest_data.add_node(word2)
    output = []

    while not source_data.is_finished() and not dest_data.is_finished():
        collision = search_level(word_dict, source_data, dest_data)
        if collision is not None:
            output = merge_paths(source_data, dest_data, collision)
        
        collision = search_level(word_dict, dest_data, source_data)
        if collision is not None:
            output = merge_paths(source_data, dest_data, collision)
    
    if len(output) == 0:
        print("No transformation is possible!")
    else:
        print(' -> '.join(output))
    
if __name__ == "__main1__":
    word_transformer_dfs(['DAMP', 'LAMP', 'LIKE', 'LIME', 'LIMP', 'HIKE', 'MIME'], 'DAMP', 'HIKE')
    word_transformer_dfs(['ALL', 'ILL', 'AIL', 'APE', 'ALE'], 'ALL', 'APE')
    word_transformer_bfs(['ALL', 'ILL', 'AIL', 'APE', 'ALE'], 'ALL', 'APE')
    word_transformer_bfs(['DAMP', 'LAMP', 'LIKE', 'LIME', 'LIMP', 'HIKE', 'MIME'], 'DAMP', 'HIKE')
    word_transformer_bi_directional_bfs(['DAMP', 'LAMP', 'LIKE', 'LIME', 'LIMP', 'HIKE', 'MIME'], 'DAMP', 'HIKE')


# Problem 17.23: Max Square Matrix: Imagine you have a square matrix, where each cell (pixel) is either black or
# white. Design an algorithm to find the maximum subsquare such that all four borders are filled with
# black pixels.
'''
    Algorithm:
        1. Square should have all the border cells value as 1.
        2. Brute Force is to try all the possible squares in O(N^4) time.
        3. We can precompute the zeros down and zeros right at each cell:
            if value is 0 then final value is 0,0
            else zeros_down = zeros_down of [row+1][col], similar logic for zeros right.
        4. So, everything will be same but validation of every square can be done in O(1) time now
        which will reduce the time to O(N^3).
        5. Important: On a Edge of Length N matrix, there are N - size + 1 squares of length size.
        6. Validation Condition: 
            top_left.zeros_right < size or top_left.zeros_down < size 
            or top_right.zeros_down < size or bottom_left.zeros_right < size: then False else True.
'''
class SquareCell(object):
    def __init__(self, zeros_down=0, zeros_right=0):
        self.zeros_right = zeros_right
        self.zeros_down = zeros_down
    def __repr__(self):
        return f'{self.zeros_down},{self.zeros_right}'

class SubSquare(object):
    def __init__(self, row, col, size):
        self.row = row
        self.col = col
        self.size = size
    def __repr__(self):
        return f'Row: {self.row}, Col: {self.col} with size: {self.size}'
    
def max_square_matrix(matrix):
    def is_valid(matrix, row, col, size, hash_matrix):
        top_left = hash_matrix[row][col]
        top_right = hash_matrix[row][col+size-1]
        bottom_left = hash_matrix[row+size-1][col]
        if (top_left.zeros_right < size or top_left.zeros_down < size 
            or top_right.zeros_down < size or bottom_left.zeros_right < size):
            return False
        return True
    
    def find_square_with_size(matrix, hash_matrix, size):
        # On a Edge of Length N matrix, there are N - size + 1 squares of length size.
        count = len(matrix) - size + 1
        for i in range(count):
            for j in range(count):
                if is_valid(matrix, i, j, size, hash_matrix):
                    return SubSquare(i, j, size)
    
    m = len(matrix)
    n = len(matrix[0])
    hash_matrix = [[None for x in range(n)] for x in range(m)]
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            if matrix[i][j] == 0:
                hash_matrix[i][j] = SquareCell(0,0)
            else:
                zeros_down = 1 if i+1 >= m else hash_matrix[i+1][j].zeros_down + 1
                zeros_right = 1 if j+1 >= n else hash_matrix[i][j+1].zeros_right + 1
                hash_matrix[i][j] = SquareCell(zeros_down, zeros_right)
    # for r in hash_matrix:
    #     for j in r:
    #         print(j, end=' ')
    #     print('\n')
    # return

    for i in reversed(range(1, m+1)):
        sub_square = find_square_with_size(matrix, hash_matrix, i)
        if sub_square is not None:
            print(sub_square)
            return

if __name__ == "__main1__":
    matrix = [
        [1,1,1,1],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,1,1]
    ]
    max_square_matrix(matrix)


# Problem 17.24: Max Submatrix: Given an NxN matrix of positive and negative integers, write code to find the
# submatrix with the largest possible sum.
'''
    Algorithm:
        1. This problem can be solved by a mix of Brute Force and some pre computing.
        2. Brute Force is to check all possible sub matrix in O(N^4) time.
        3. Optimized part is to pre calculate the value of area from origin to that cell by following
        DP formula:
            A[i][j] = A[i-1][j] + A[i][j-1] - A[i-1][j-1] + value[i][j] [Pretty straight forward after visulaizing it]
        Once done, sum of every possible matrix can be computed in O(1) time by following formula:
            Area of matrix ranging from r1, c1 to r2, c2:
                Area = area_matrix[r2][c2] - area_matrix[r2][c1-1] - area_matrix[r1-1][c2] + area_matrix[r1-1][c1-1]
                [This is also clear after the visulization]
        4. Keep updating the max sum and max sum submatrix as we traverse through the matrix.
        5. Overall time O(N^4).
'''
def max_sum_submatrix(matrix):
    def build_area_matrix(matrix, area_matrix):        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                left = 0 if j-1 < 0 else area_matrix[i][j-1]
                top = 0 if i-1 < 0 else area_matrix[i-1][j]
                common = area_matrix[i-1][j-1] if i-1 >= 0 and j-1 >= 0 else 0
                area_matrix[i][j] = left + top - common + matrix[i][j]

    def get_sum(area_matrix, r1, c1, r2, c2):
        left = area_matrix[r2][c1-1] if c1 > 0 else 0
        top = area_matrix[r1-1][c2] if r1 > 0 else 0
        top_and_left = area_matrix[r1-1][c1-1] if r1 > 0 and c1 > 0 else 0
        return area_matrix[r2][c2] - left - top + top_and_left
                
    m = len(matrix)
    n = len(matrix[0])
    area_matrix = [[None for x in range(n)] for x in range(m)]
    build_area_matrix(matrix, area_matrix)

    max_sum = 0
    submatrix = [0]
    for r1 in range(m):
        for r2 in range(m):
            for c1 in range(n):
                for c2 in range(n):
                    l_sum = get_sum(area_matrix, r1, c1, r2, c2)
                    if l_sum > max_sum:
                        max_sum = l_sum
                        submatrix[0] = (r1, c1, r2, c2)
    print(f'Max Sum is: {max_sum}')
    print(f'Sub Matrix is: R1:{submatrix[0][0]} C1:{submatrix[0][1]} R2:{submatrix[0][2]} C2:{submatrix[0][3]}')

if __name__ == "__main1__":
    matrix = [
        [-8,1,1,1],
        [1,1,1,1],
        [1,1,-5,1],
        [1,1,1,1]
    ]
    max_sum_submatrix(matrix)


# Problem 17.25: Word Rectangle: Given a list of millions of words, design an algorithm to create the largest possible
# rectangle of letters such that every row forms a word (reading left to right) and every column forms
# a word (reading top to bottom). The words need not chosen consecutively from the list, but all
# rows must be the same length and all columns must be the same height.
'''
    Algorithm:
        1. This problem is both complex and its complete implementation is also very lengthy.
        2. Key points in the algorithm:
            a. Group the words of similar length words in a separate dictionary (imagine nested dictionary).
            b. We will start creating rectangles from maximum word lengths.
            c. Brute Force approach would be, if we are trying to create a rectangle of 5x4 then try
            all sequences of words of length 5 and then validate each rectangle (by checking each column is a word
            or not) which will surely take a lot of time.
            d. We can implement a Trie for each column check and as soon as we find that column is not a valid 
            prefix then we discard the rectangle.
            e. We do not need to create the Tries for all possible heights in advance, we will create them as and
            when we need them.
'''
class Rectangle(object):
    def __init__(self, h=None, l=None, word_matrix=None):
        self.h = h
        self.l = l
        self.word_matrix = word_matrix
    def isComplete(self, trie_list):
        pass
    def is_partial_ok(self, trie_list):
        pass

class TrieNode2(object):
    def __init__(self):
        self.children = [None] * 26
        self.isLeaf = False

class Trie2(object):
    def __init__(self):
        self.root = TrieNode2()
    def insert_word(self, word):
        pass
    def search_prefix(self, prefix):
        pass
    def search_word(self, word):
        pass

def word_rectangle(word_dict):
    def group_word_lengths(word_dict):
        pass

    def create_trie(word_dict, h, trie_list):
        trie_list[h - 1] = Trie()
        pass

    def make_rectangle(word_dict, l, h, trie_list):
        # will be called recursively
        if trie_list[h - 1] is None:
            create_trie(word_dict, h, trie_list)
        return make_partial_rectangle(word_dict, l, h, trie_list, None)

    def make_partial_rectangle(word_dict, l, h, trie_list, rectangle):
        if rectangle is None:
            rectangle = Rectangle()
        if rectangle.h == h:
            if rectangle.isComplete(trie_list):
                return rectangle
            return
        if not rectangle.is_partial_ok(trie_list):
            return
        for i in range(l):
            make_partial_rectangle(word_dict, i, h, trie_list, rectangle)
    
    group_word_lengths(word_dict)
    # Maximum word length
    max_length = max([x for x in word_dict])
    trie_list = [None for x in range(max_length)]
    for z in reversed(range(1, max_length + 1)):
        for i in reversed(range(1, max_length)):
            rectangle = make_rectangle(word_dict, i, j, trie_list)
            if rectangle is not None:
                return rectangle
            

# Problem 17.26: Sparse Similarity: The similarity of two documents (each with distinct words) is defined to be the
# size of the intersection divided by the size of the union. For example, if the documents consist of
# integers, the similarity of {1, 5, 3} and {1, 7, 2, 3} is 0.4, because the intersection has size
# 2 and the union has size 5.
# We have a long list of documents (with distinct values and each with an associated ID) where the
# similarity is believed to be "sparse:'That is, any two arbitrarily selected documents are very likely to
# have similarity 0. Design an algorithm that returns a list of pairs of document IDs and the associated
# similarity.
# Print only the pairs with similarity greater than 0. Empty documents should not be printed at all. For
# simplicity, you may assume each document is represented as an array of distinct integers.
# EXAMPLE
# Input:
#   13: {14, 15, 100, 9, 3}
#   16: {32, 1, 9, 3, 5}
#   19: {15, 29, 2, 6, 8, 7}
#   24: {7, 10}
# Output:
#   ID1, ID2:   SIMILARITY
#   13, 19  :   0.1    
#   13, 16  :   0.25
#   19, 24  :   0.14285714285714285