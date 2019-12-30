# Egg Drop Problem: Find the minimum number of Drops required to find out the floor from which an egg starts dropping
# with n Eggs and k floors.
def egg_drop(n, k):
    '''
        Algorithm:
            1. If n = 1, drops will be same as number of floors. Also, if k = 1 then also drops will be 1.
            2. If we drop an egg from floor x then 2 cases can happen:
                a. If it breaks then we need to find out the drops with n-1 eggs and x-1 floors.
                b. If it doesn't break then we need to find out the drops with n eggs and k-x floors.
                c. Maximum of both these cases will be the required number of drops needed if we start dropping an
                egg from floor number x.
            3. So, if we repeat the whole steps for every possible x, we will get the minimum drops.
            4. Since, picking a floor matters so we can easily track the starting floor.
            5. Code involves a lot of recursion so memoization has to be done.
    '''
    def egg_drop_util(n, k, floor, cache):
        if n == 1 or k == 0 or k == 1: return k
        try:
            val =  cache[str(n) + ':' + str(k)]
            return val
        except:
            None
        min_drops = k
        floor_num = 0
        for x in range(1, k + 1):
            drops = max(egg_drop_util(n - 1, x - 1, floor, cache), egg_drop_util(n, k - x, floor, cache))
            if drops < min_drops:
                min_drops = drops
                floor[0] = x
        cache[str(n) + ':' + str(k)] = min_drops + 1
        return min_drops + 1
    
    if n == 0: return 
    floor = [-1]
    cache = {}
    result = egg_drop_util(n, k, floor, cache)
    print('Minimum Drops Needed: ' + str(result))
    print('Floor Number to Start: ' + str(floor[0]))

# find the length of the longest subsequence of a given sequence such that all elements of the subsequence are sorted in increasing order.
def lis_brute_force(arr):
    results = [0]
    max_length = 0
    for r in range(len(arr)):
        result = []
        result.append(arr[r])
        for r1 in range(r + 1, len(arr)):
            if arr[r1] > result[len(result) - 1]:
                result.append(arr[r1])
        if len(result) > max_length:
            results[0] = list(result)
            max_length = len(result)
    print(f'LIS is: {results[0]}')
    print(f'LIS Length is: {len(results[0])}')

def lengthOfLIS(arr):
    '''
        Algorithm:
            1. This one is the most efficient algorithm to find out the LIS Lenght.
            2. It can only find out the LIS Lenght and not LIS.
            3. To implement it, we need to implement Arrays.binary_search method from Java that is a binary serach 
            method that takes array, value, low and high indes. It returns index if value is found else
            -(insertion_point + 1) where insertion_point is the index where value should be present. 
            4. We will create an array of same length as input array and initialize this with some negative values
            that are not present in input array.
            5. For each element in input array, adjust the element in placeholder by either inserting it in the end
            or in middle.
            6. Each time we insert the element in the end of the placeholder, result will be increased by 1.
            7. Final placeholder array wont be the LIS but its length will always be same as LIS.
    '''
    def binary_search(arr, value, low, high):
        def binary_search_util(arr, value, low, high, insertion_point):
            if low > high:
                return -1
            mid = (low + high) // 2
            if arr[mid] == value:
                return mid
            elif arr[mid] > value:
                if mid < insertion_point[0]:
                    insertion_point[0] = mid
                return binary_search_util(arr, value, low, mid - 1, insertion_point)
            else:
                return binary_search_util(arr, value, mid + 1, high, insertion_point)

        insertion_point = [high + 1]
        result =  binary_search_util(arr, value, 0, high, insertion_point)
        if result == -1:
            return - insertion_point[0] - 1
        else:
            return result

    dp = [-1 for x in arr]
    result = 0
    for r in arr:
        i = binary_search(dp, r, 0, result-1)
        if i < 0:
            i = - (i + 1)
        dp[i] = r
        if i == result:
            result += 1
    print(f'Final Placeholder Array looks like: {dp}')
    print(f'Length of LIS is: {result}')    


if __name__ == "__main__":
    print('Egg Drop Problem:')
    egg_drop(2, 100) 

    print()
    print("Longest Increasing Subsequence in increasing order via Brute Force:")
    arr = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    lis_brute_force(arr)