from collections import Counter

# Problem:1.1 Is Unique: Implement an algorithm to determine if a string has all unique characters. What if you
# cannot use additional data structures?
def is_unique_no_storage(string):
    '''
    Algorithm: If we can't use additional Data Structure then we simply need to check each charcter against all other characters.
    '''
    n = len(string)
    for i in range(n):
        for j in range(i+1, n):
            if string[j] == string[i]:
                return False
    return True

def is_unique_storage(string):
    '''
    Assumption: String only has ASCII characters. 
    Algorithm: Create a list of length 128 and set each element to 0. For each character in String, take its ord/ASCII value and check the same index in list. If value is 1 then return False else set that value to 1. If loop completes then return True.
    '''
    char_list = [0 for x in range(128)]
    for r in string:
        if char_list[ord(r)] == 1:
            return False
        else:
            char_list[ord(r)] = 1
    return True
    

# Problem:1.2 Check Permutation: Given two strings, write a method to decide if one is a permutation of the
# other.
def is_permutation(s1, s2):
    '''
    Algorithm: If 2 strings are permutations of each other then they must have same characters with same repetitions.
    We just need to compare the sorted string to each as they must be equal if strings are permutation of each
    other.
    '''
    return sorted(s1) == sorted(s2)


# Problem:1.3 URLify: Write a method to replace all spaces in a string with '%20'. You may assume that the string
# has sufficient space at the end to hold the additional characters, and that you are given the "true"
# length of the string. (Note: If implementing in Java, please use a character array so that you can
# perform this operation in place.)
# EXAMPLE
# Input: "Mr John Smith  ", 13
# Output: "Mr%20John%20Smith"
def urlify(string, true_length):
    '''
    Algorithm: If s is 'a b c    ' then n will be 5.
    If the True Length of the string is given then this can be done in-place.
    We will run a reversed loop on the True Length. 
    Start the write index from end of the string. If loop index is space, write %20 in the end and subtract 
    write index by 3 else write loop index value in the end and subtract write index by 1.
    '''
    if string is None: return
    result = list(string)
    index = len(result) - 1
    for r in reversed(range(true_length)):
        if result[r] == ' ':
            result[index] = '0'
            result[index - 1] = '2'
            result[index - 2] = '%'
            index -= 3
        else:
            result[index] = result[r]
            index -= 1
    return ''.join(result)


# Problem:1.4 Palindrome Permutation: Given a string, write a function to check if it is a permutation of a palindrome.
# A palindrome is a word or phrase that is the same forwards and backwards. A permutation
# is a rearrangement of letters. The palindrome does not need to be limited to just dictionary words.
# EXAMPLE
# Input: Tact Coa
# Output: True (permutations: "taco cat", "atco eta", etc.)
def is_palindrome_permutation(s):
    '''
    Algorithm: A String can be a palindrome in following cases:
        1. If its length is even then all letters should be in pair.
        2. If its length is odd then exactly 1 letter should be without a pair.
    Create a counter from input string. From this counter, we will create a list only if value of the key is not even.
    If string length is even then list will have 0 elements else 1 element. Other cases will be False.
    '''
    c = Counter(s)
    l = [x for x in c.keys() if c[x] % 2 != 0]
    return len(l) <= 1


# Problem:1.5 One Away: There are three types of edits that can be performed on strings: insert a character,
# remove a character, or replace a character. Given two strings, write a function to check if they are
# one edit (or zero edits) away.
# EXAMPLE
# pale, ple -> true
# pales, pale -> true
# pale, bale -> true
# pale, bake -> false
def is_one_edit_away(s1, s2):
    '''
    Algorithm: 
        Find out the shorter and longer strings, start the index1 and index2 for both the strings.
        Loop through the strings till index1 or index2 reaches their max limit.
        A variable diff will be set to False.
        If index value is same for both the strings, increase index1 and index2 by 1.
        else:
            if diff is already True then return False.
            if length are same then increase both the indexes else increase only index 2 (longer string).
            set the diff to True.
        return True after the loop.
    '''
    if s1 is None or s2 is None: return     
    l1 = len(s1)
    l2 = len(s2)
    if abs(l1 - l2) > 1: return False
    shorter = s1 if l1 <= l2 else s2
    longer = s2 if l1 <= l2 else s1

    index1 = index2 = 0
    diff = False
    while index1 < l1 and index2 < l2:
        if shorter[index1] == longer[index2]:
            index1 += 1
        else:
            if diff:
                return False
            else:
                diff = True
                if l1 == l2: index1 += 1        
        index2 += 1
    return True


# Problem:1.6 String Compression: Implement a method to perform basic string compression using the counts
# of repeated characters. For example, the string aabcccccaaa would become a2b1c5a3. If the
# "compressed" string would not become smaller than the original string, your method should return
# the original string. You can assume the string has only uppercase and lowercase letters (a - z).
def string_compression(string):
    '''
    Algorithm:
    No tricks here, a standard traversal of array and variable tracking.
    '''
    if string is None: return
    result = []
    index = 0
    
    while index < len(string):
        value = string[index]
        result.append(value)
        count = 1
        for j in range(index + 1, len(string)):
            if string[j] == value:
                count += 1
                index += 1
            else:
                break            
        index += 1
        if count > 1: result.append(str(count))

    return ''.join(result)


# Problem:1.7 Rotate Matrix: Given an image represented by an NxN matrix, where each pixel in the image is 4
# bytes, write a method to rotate the image by 90 degrees. Can you do this in place?
def rotate_matrix_inplace(matrix):
    '''
    Algorithm: If rotation can be done by creating another matrix then it can be done by traversing the matrix linearly.
    To do it in place, we just need to go level by level. If matrix length is 4 or 5 then 2 level needs to be traversed.
    Level 1 is outer rows, level 2 will be inner rows. Level 3 will only have 1 cell so rotation is not applicable.
    Once level is determined, we need to rotate each cell by creating a temp variable.
    Hints:
        Loop will run very less number of times. In a 4x4 matrix, loop will run 3 times for level 0.
	    4 replacement will be done.
		For each replacement, think what is constant: row or column and then build the logic.
    '''
    n = len(matrix)
    levels = n // 2
    for level in range(levels):
        start = level
        end = n - level - 1
        for r in range(start, end):
            temp = matrix[level][r]
            matrix[level][r] = matrix[-r -1][level]
            matrix[-r -1][level] = matrix[-level -1][-r -1]
            matrix[-level -1][-r -1] = matrix[r][-level -1]
            matrix[r][-level -1] = temp
    for r in matrix:
        print(r)


# Problem:1.8 Zero Matrix: Write an algorithm such that if an element in an MxN matrix is 0, its entire row and
# column are set to 0.
def zero_matrix(matrix):
    '''
    Algorithm: Straight forward matrix traversal. To save the work we can store rows and columns in a set.
    '''
    def set_row(row):
        for r in range(len(matrix[0])):
            matrix[row][r] = 0
    def set_col(col):
        for r in range(len(matrix)):
            matrix[r][col] = 0
    rows = set()
    cols = set()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                rows.add(i)
                cols.add(j)
    for r in rows:
        set_row(r)
    for c in cols:
        set_col(c)


# Problem:1.9 String Rotation:Assume you have a method isSubstring which checks if one word is a substring
# of another. Given two strings, sl and s2, write code to check if s2 is a rotation of s1 using only one
# call to isSubstring (e.g., "waterbottle" is a rotation of"erbottlewat").
def is_rotation(s1, s2):
    '''
    Algorithm:
    if s1 is a rotation of s2 then s1 will be a substring of s2s2.
    Ex: s1 = xyz, s2 = zxy then s2s2 = zxyzxy
    We can use "s1 in s2" statement to mimic substring method.
    '''
    def is_substring(s1, s2):
        return s1 in s2
    return is_substring(s1, s2+s2)


if __name__ == "__main__":
    print("Problem# 1.1")
    print(is_unique_no_storage('bcada'))
    print(is_unique_storage('abc'))

    print("\nProblem# 1.2")
    print(is_permutation('abac', 'cbaa'))

    print("\nProblem# 1.3")
    print(urlify('Mr John Smith    ', 13))

    print("\nProblem# 1.4")
    print(is_palindrome_permutation('assac'))

    print("\nProblem# 1.5")
    print(is_one_edit_away('dal', 'pal'))   

    print("\nProbelm# 1.6")     
    print(string_compression('abbbbbcccccd'))

    print("\nProblem 1.7")
    # m = [
    # [1,2,3],
    # [4,5,6],
    # [7,8,9]
    # ]
    m = [
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
    ]
    for r in m:
        print(r)
    print()
    rotate_matrix_inplace(m)

    print("\nProblem# 1.8")
    m = [
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
    ]
    zero_matrix(m)
    for r in m:
        print(r)

    print("\nProblem# 1.9")
    print(is_rotation('abcdef', 'bcdefa'))