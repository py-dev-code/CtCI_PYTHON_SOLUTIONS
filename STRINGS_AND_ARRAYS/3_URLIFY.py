def urlify(str, length):
    # str will be the complete string with enough amount of spaces.
    # lenght is the length of string with content
    # output will be a string with all spaces updated to %20 in input string
    new_index = len(str)
    str = list(str) # Changing string to list as String is immutable in Python
    for r in reversed(range(length)):
        if str[r] == ' ':
            str[new_index - 3: new_index] = '%20'
            new_index = new_index - 3
        else:
            str[new_index - 1] = str[r]
            new_index = new_index - 1

    return ''.join(str)

print(urlify('a b c  d        ',8))

