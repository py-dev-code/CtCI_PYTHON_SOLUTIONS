def get_n_factorial(n, l=None):
    '''
    A function to calculate Factorial of given number by using memoization.
    This is an example of Top-Down approach.
    '''
    if l is None:
        l = [None for x in range(n+1)]
    if n < 0:
        return None
    elif n == 0:
        return 1
    elif l[n] is not None:
        return l[n]
    else:
        l[n] = n * get_n_factorial(n-1, l)
        return l[n]


def get_n_factorial_bottom_up(n):
    '''
    Program to calculate n factorial with bottom up approach.
    '''
    if n < 0:
        return
    l = [None for x in range(n+1)]
    l[0] = 1
    for r in range(1, n+1):
        l[r] = r * l[r-1]
    return l[n]


def get_nth_fibonacci_num(n, l=None):
    '''
    A function to calculate nth Fibonacci Number by using memoization.    
    '''
    if l is None:
        l = [None for x in range(n+1)]
    if n < 0:
        return
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    elif l[n] is not None:
        return l[n]
    else:
        l[n] = get_nth_fibonacci_num(n-1, l) + get_nth_fibonacci_num(n-2, l)
        return l[n]


def get_nth_fibonacci_num_bottom_up(n):
    '''
    Function to get nth fibonacci number by using bottom up approach.
    '''
    if n < 0:
        return
    # Using max so that we can handle input 0 and 1.
    l = [None for x in range(max(2, n+1))]
    l[0] = 0
    l[1] = 1
    for r in range(2, n+1):
        l[r] = l[r-1] + l[r-2]
    return l[n]


if __name__ == "__main__":
    print(get_n_factorial(10))                  # 3628800
    print(get_n_factorial_bottom_up(10))        # 3628800
    print(get_nth_fibonacci_num(10))            # 55
    print(get_nth_fibonacci_num_bottom_up(10))  # 55
