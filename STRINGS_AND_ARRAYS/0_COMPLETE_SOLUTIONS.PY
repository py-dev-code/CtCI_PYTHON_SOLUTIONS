def is_string_rotated(str1, str2):
    return str2 in str1+str1
# print(is_string_rotated('abcd','dabc'))

def zero_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    zero_rows = []
    zero_cols = []
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                if i not in zero_rows: zero_rows.append(i)
                if j not in zero_cols: zero_cols.append(j)
    for i in zero_rows:
        for j in range(cols):
            matrix[i][j] = 0
    for i in zero_cols:
        for j in range(rows):
            matrix[j][i] = 0
    for i in range(rows):
        for j in range(cols):
            print(matrix[i][j], end = ' ')
        print()
# matrix = [
#     [1,2,3,0],
#     [1,0,3,4],
#     [1,2,3,4]
# ]
# zero_matrix(matrix)

def rotate_matrix_90_degree(matrix):
    if len(matrix) != len(matrix[0]): return "Matrix is not a Square"
    for layer in range(len(matrix) // 2):
        start = layer
        end = len(matrix) - layer - 1
        # print('{} - {}:{}'.format(layer, start, end))
        for i in range(start, end):
            top = matrix[layer][i]
            matrix[layer][i] = matrix[- i - 1][layer]            
            matrix[- i - 1][layer] = matrix[- layer - 1][- i - 1]            
            matrix[- layer - 1][- i - 1] = matrix[i][- layer - 1]            
            matrix[i][- layer - 1] = top
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end = ' ')        
        print()    
# matrix = [
#     [1,2,3,10],
#     [4,5,6,11],
#     [7,8,9,12],
#     [13,14,15,16]
# ]
# rotate_matrix_90_degree(matrix)

def is_one_away_book(str1, str2):
    n1 = len(str1)
    n2 = len(str2)
    i, j = 0, 0
    if abs(n1 - n2) > 1: return False    
    edited = False
    if n1 == n2:
        for i, j in zip(str1, str2):
            if i != j:
                if edited:
                    return False
                edited = True
    if n1 < n2: 
        while i < n1 and j < n2:
            if str1[i] != str2[j]:
                if edited:
                    return False
                edited = True
                j += 1
            else:
                i += 1
                j += 1
    if n1 > n2: 
        while i < n1 and j < n2:
            if str1[i] != str2[j]:
                if edited:
                    return False
                edited = True
                i += 1
            else:
                i += 1
                j += 1
    return True

def is_one_away_mine(s1, s2):
    n1, n2 = len(s1), len(s2)
    if abs(n1 - n2) > 1: return False
    i, j = 0, 0
    edit_count = 0
    while i < n1 and j < n2:
        if s1[i] != s2[j] and n1 == n2:
            edit_count += 1
        elif s1[i] != s2[j] and n1 < n2:
            edit_count += 1
            i -= 1
        elif s1[i] != s2[j] and n2 > n2:
            edit_count += 1
            j -= 1
        i += 1
        j += 1
    if edit_count > 1: return False
    return True

# print(is_one_away_mine('abc','adc'))
# print(is_one_away_mine('abcs','abc'))
# print(is_one_away_mine('abc','abcs'))

def is_palindrome_permutation(s):
    char_list = [0 for _ in range(ord('z') - ord('a') + 1)]
    count_odd = 0
    for chr in s:
        if ord(chr) not in range(ord('a'), ord('z') + 1): continue
        x = ord(chr) - ord('a') 
        char_list[x] += 1
        if char_list[x] % 2:
            count_odd += 1
        else:
            count_odd -= 1
    return count_odd <= 1
# print(is_palindrome_permutation('aabbc_='))

def is_string_anagram(s1, s2):
    return ''.join(sorted(s1)) == ''.join(sorted(s2))
# print(is_string_anagram('abcdef', 'bcedfa'))

def is_unique(s):
    char_list = [False for _ in range(0, 128)]
    for chr in s:
        if char_list[ord(chr)]: return False
        else: char_list[ord(chr)] = True
    return True
# print(is_unique('unqQ'))

def urlify(s, n):
    new_index = len(s)
    s = list(s)
    for r in reversed(range(n)):
        if s[r] == ' ':
            s[new_index - 3: new_index] = '%20'
            new_index -= 3
        else:
            s[new_index - 1] = s[r]
            new_index -= 1
    return ''.join(s)
# print(urlify('a b c    ', 5))






    
        


        




