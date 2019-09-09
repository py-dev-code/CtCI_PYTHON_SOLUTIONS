def fact_top_down(n):
    if n == 0:
        return 1
    if f[n] != -1:
        return f[n]
    f[n] = n * fact_top_down(n - 1)
    return f[n]
# f = [-1 for _ in range(0, 101)]
# result = fact_top_down(6)
# print(result)

def fibo_memoization(n):    
    # global count
    if n == 0:
        return 0
    if n == 1:
        return 1
    if f[n] != -1:
        # count += 1
        # print(count)
        return f[n] 
    f[n] = fibo_memoization(n - 1) + fibo_memoization(n - 2)
    return f[n]
f = [-1 for _ in range(1, 500)]
count = 0
result = fibo_memoization(80)
print(result)