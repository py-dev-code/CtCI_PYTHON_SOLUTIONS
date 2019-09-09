# Egg Drop Problem: Find the minimum number of drops needed for 2 eggs to find out 
# from which floor does an egg start breaking given f floors.

# Below is the worst performing code. It will take like 5 min for 20 floors
def egg_drop(n, f):
    if f == 0 or f == 1: return f
    if n == 1: return f
    min = 999
    for x in range(1, f + 1):
        val = max(egg_drop(n - 1, x - 1), egg_drop(n, f - x))
        if val < min:
            min = val    
    return min + 1
# s = egg_drop(2, 10)    
# print(s)
    
# Approach with memoization
def egg_drop_memoization(n, k):
    egg_floor_list = [[0 for x in range(k + 1)] for x in range(n + 1)] 
    # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for i in range(1, n + 1):
        egg_floor_list[i][1] = 1
        egg_floor_list[i][0] = 0       
    # print(egg_floor_list)
    for j in range(1, k + 1):
        egg_floor_list[1][j] = j
    # print(egg_floor_list)
    
    for i in range(2, n + 1):
        for j in range(2, k + 1):
            egg_floor_list[i][j] = INT_MAX
            for x in range(1, j + 1):
                res = 1 + max(egg_floor_list[i - 1][x - 1], egg_floor_list[i][j - x])
                if res < egg_floor_list[i][j]:
                    egg_floor_list[i][j] = res
    return egg_floor_list[i][j]

INT_MAX = 32767    
val = egg_drop_memoization(2, 100)
print(val)

    
