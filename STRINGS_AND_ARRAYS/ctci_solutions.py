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
# Input: "Mr John Smith ", 13
# Output: "Mr%20John%20Smith"
def urlify(s, n):
    '''
    Algorithm: If s is 'ab c  ' then n will be 4.
    We will create a list with same length as s.
    Create an index variable with 0 and loop through s and build the final string.
    '''
    result = [' ' for x in s]
    index = 0
    for r in range(n):
        if s[r] == ' ':
            result[index] = '%'
            result[index + 1] = '2'
            result[index + 2] = '0'
            index += 3
        else:
            result[index] = s[r]
            index += 1
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
    if len(s) % 2 == 0:
        return len(l) == 0
    else:
        return len(l) == 1


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
    Algorithm: Convert both strings into arrays.
    The idea is that based on lenght of the arrays, we will make 1 edit in one of the array. After the 1st edit, 
    if strings of both arrays are same then return true else return false.
    Edit Decision:
    If length is same then update element to other array on 1st mismatch.
    If length is different then delete the element from larger array on 1st mismatch.
    '''
    if s1 == s2:
        return True
    l1 = list(s1)
    l2 = list(s2)
    n1, n2 = len(l1), len(l2)
    
    if n1 == n2:
        # Update case
        for r in range(n1):
            if l1[r] != l2[r]:
                l1[r] = l2[r]
                break
    elif n1 > n2:
        # Delete case from l1
        for r in range(n2):
            if l1[r] != l2[r]:                
                l1.remove(l1[r])
                break
            elif r == n2 - 1:
                l1.remove(l1[r + 1])                
    elif n1 < n2:
        # Delete case from l2
        for r in range(n1):
            if l1[r] != l2[r]:
                l2.remove(l2[r])
                break
            elif r == n1 - 1:
                l2.remove(l2[r + 1])
    return ''.join(l1) == ''.join(l2)


# Problem:1.6 String Compression: Implement a method to perform basic string compression using the counts
# of repeated characters. For example, the string aabcccccaaa would become a2b1c5a3. If the
# "compressed" string would not become smaller than the original string, your method should return
# the original string. You can assume the string has only uppercase and lowercase letters (a - z).
def string_compression(s):
    '''
    Algorithm:
    No tricks here, a standard traversal of array and variable tracking.
    '''
    result = []
    result.append(s[0])
    compress = 1
    last_char = s[0]
    for r in range(1, len(s)):
        if s[r] == last_char:
            compress += 1
        elif s[r] != last_char and compress > 1:
            last_char = s[r]
            result.append(str(compress))
            result.append(last_char)
            compress = 1
        elif s[r] != last_char and compress == 1:
            last_char = s[r]
            result.append(last_char)
    if compress > 1:
        result.append(str(compress))
    return ''.join(result)


# Problem:1.7 Rotate Matrix: Given an image represented by an NxN matrix, where each pixel in the image is 4
# bytes, write a method to rotate the image by 90 degrees. Can you do this in place?
def rotate_matrix_inplace(matrix):
    '''
    Algorithm: If rotation can be done by creating another matrix then it can be done by traversing the matrix linearly.
    To do it in place, we just need to go level by level. If matrix length is 4 or 5 then 2 level needs to be traversed.
    Level 1 is outer rows, level 2 will be inner rows. Level 3 will only have 1 cell so rotation is not applicable.
    Once level is determined, we need to rotate each cell by creating a temp variable.
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
            # break
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
# of another. Given two strings, sl and s2, write code to check if s2 is a rotation of sl using only one
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
    print(urlify('a b c    ', 5))

    print("\nProblem# 1.4")
    print(is_palindrome_permutation('assac'))

    print("\nProblem# 1.5")
    print(is_one_edit_away('dal', 'pal'))   

    print("\nProbelm# 1.6")     
    print(string_compression('abbbbbcccccd'))

    print("\nProblem 1.7")
    m = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
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
    print(is_rotation('abcdef', 'cdefba'))