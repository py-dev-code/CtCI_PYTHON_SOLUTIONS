def check_permutation(str1, str2):
    # Logic: check the length first. Then, compare the sorted strings
    if len(str1) != len(str2):
        return False
    # sorted() returns a list so converting the list into string by join()
    return "".join(sorted(str1)) == "".join(sorted(str2))   

print(check_permutation("aaja", "ajaa"))
print(check_permutation("aaja", "bjaa"))