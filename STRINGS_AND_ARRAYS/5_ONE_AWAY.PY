def isOneAway(str1, str2):
    
    len1 = len(str1)
    len2 = len(str2)

    editCount = 0
    idx = 0

    # Length Difference can not be more than 1
    if (abs(len1 - len2) > 1):
        return False

    for r in range(min(len1, len2)):
        if (str1[r] != str2[idx] and len1 == len2):
            editCount += 1    
        elif (str1[r] != str2[idx] and len1 > len2):
            editCount += 1
            idx -= 1
        elif (str1[r] != str2[idx] and len1 < len2):
            editCount += 1
            idx += 1
        idx += 1

    if editCount > 1:
        result = False
    else:
        result = True

    return result

print(isOneAway('pal','palks'))