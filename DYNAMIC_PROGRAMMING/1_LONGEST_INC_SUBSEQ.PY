# Program to find the lenght of longest increasing sequence. Elements do not need to be adjacent.
global maximum
maximum = 1
def lis(arr):
    global maximum
    n = len(arr)
    _lis(arr, n)
    print(l)
    return maximum
def _lis(arr, n):
    if n == 1: return 1
    if l[n - 1] != 0: return l[n - 1]
    global maximum
    max_ending_here = 1
    for i in range(1, n):
        res = _lis(arr, i)
        if arr[i - 1] < arr[n - 1] and res + 1 > max_ending_here:
            max_ending_here = res + 1
    maximum = max(maximum, max_ending_here)
    l[n - 1] = maximum
    return max_ending_here
arr = [10, 22, 30, 9, 33, 21, 50, 41, 60, 80, 90, 1,2,3,4,5,6]
l = [0 for x in arr]
print(l)

print(lis(arr))
