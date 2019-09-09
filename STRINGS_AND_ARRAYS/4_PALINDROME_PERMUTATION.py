def pal_perm(str):
    '''
    Logic: Remove all the pair from String then either 1 character should be left [if string is odd] or 
    no character should be left [if string is even].
    Make a list of all 0s for all ASCII lowercase characters. Keep adjusting the count variable in the loop.
    In the end it should be less than or equal to 1.
    '''
    char_list = [0 for _ in range(ord('z') - ord('a') + 1)]
    countodd = 0    
    for c in str:
        
        # Ignoring characters not in a to z
        if (ord(c) not in range(ord('a'), ord('z') + 1)):
            continue

        x = ord(c) - ord('a')
        char_list[x] += 1
        if char_list[x] % 2:
            countodd += 1
        else:
            countodd -= 1
        
    return countodd <= 1            

print(pal_perm("assac".lower()))
