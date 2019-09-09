def isUnique(string):
    # Assuming character set is ASCII (128 characters)
    if len(string) > 128:
        return False
    
    # Creating a boolean array with all False 
    char_set = [False for _ in range(128)]

    for char in string:
        val = ord(char) # Returns the Unicode value of the character [same as ASCII]
        if char_set[val]:
            return False
        char_set[val] = True

    return True

print(isUnique("abc"))