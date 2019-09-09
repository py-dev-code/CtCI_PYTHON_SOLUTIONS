def compressed(string):

    counter = 1
    lst = []

    for r in range(1, len(string)):
        if (string[r] == string[r - 1]):
            counter += 1               
        else:
            if (counter > 1):
                lst.append(string[r - 1] + str(counter))
            else:
                lst.append(string[r - 1])    
            counter = 1 

    if (counter > 1):
        lst.append(string[-1] + str(counter))
    else:
        lst.append(string[-1])

    return ''.join(lst)    

print(compressed('abcdef'))    