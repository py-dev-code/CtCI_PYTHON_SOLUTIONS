def isRotation(str1, str2):
    return (str1 + str1).find(str2) != -1

print(isRotation('hithere', 'herehit'))
