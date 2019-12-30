def get_any_path(m):
    '''
    Problem:
        1. Only Right and Down movement is allowed.
        2. Cells with 0 value are not reachable.
    Algorithm:
        1. Start with cell [0][0]
        2. As soon as we reach in the end, return the path.        
    '''
    def get_any_path_util(m, r1, c1, r, c, path):
        if r1 > r or c1 > c or m[r1][c1] == 0:
            return False        
        path.append(f'[{r1}][{c1}]')
        if r1 == r and c1 == c:
            return True
        if get_any_path_util(m, r1+1, c1, r, c, path):
            return True
        else:
            return get_any_path_util(m, r1, c1+1, r, c, path)        

    if m is None or len(m) == 0:
        return
    path = []
    r = len(m) - 1
    c = len(m[0]) - 1
    if get_any_path_util(m, 0, 0, r, c, path):
        return path
    else:
        return "No Path Found!"

def get_all_paths(m):
    '''
    Problem:
        1. Only Right and Down movement is allowed.
        2. Cells with 0 value are not reachable.
        3. Need to print all possible paths.
    Algorithm:
        1. Start with cell [0][0]
        2. As soon as we reach in the end, append the path in paths and no return.
        3. Once both movements are done, pop the cell [r1][c1] from the path so that it can be used in other paths.
    '''
    def get_all_paths_util(m, r1, c1, r, c, path, paths):
        if r1 > r or c1 > c or m[r1][c1] == 0:
            return
        path.append(f'[{r1}][{c1}]')
        if r1 == r and c1 == c:
            paths.append(list(path))
        get_all_paths_util(m, r1+1, c1, r, c, path, paths)        
        get_all_paths_util(m, r1, c1+1, r, c, path, paths)
        path.pop()

    if m is None or len(m) == 0:
        return
    path = []
    paths = []
    r = len(m) - 1
    c = len(m[0]) - 1
    get_all_paths_util(m, 0, 0, r, c, path, paths)
    if len(paths) > 0:
        return paths
    else:
        return ["No Path Found!"]

def get_all_paths_count(m):
    '''
    Problem:
        1. Only Right and Down movement is allowed.
        2. Cells with 0 value are not reachable.
        3. Need to get the count of all possible paths.
    Algorithm:
        1. Start with cell [0][0]
        2. As soon as we reach in the end, return 1.
        3. Use a cache to store the possible paths from any given cell.
    '''
    def get_all_paths_count_util(m, r1, c1, r, c, cache):
        if r1 > r or c1 > c or m[r1][c1] == 0:
            return 0
        if cache[r1][c1] is not None:
            return cache[r1][c1]
        result = 0
        if r1 == r and c1 == c:
            return 1
        result += get_all_paths_count_util(m, r1+1, c1, r, c, cache)
        result += get_all_paths_count_util(m, r1, c1+1, r, c, cache)
        cache[r1][c1] = result
        return result
    
    if m is None or len(m) == 0:
        return 0    
    r = len(m) - 1
    c = len(m[0]) - 1
    cache = [[None for x in range(c+1)] for x in range(r+1)]
    return get_all_paths_count_util(m, 0, 0, r, c, cache)

def get_all_paths_count_all_dir(m):
    '''
    Problem:
        1. All movements are allowed.
        2. Cells with 0 value are not reachable.
        3. Need to print all possible paths.
    Algorithm:
        1. Start with cell [0][0].
        2. In this, we need to maintain a visited status for each cell so that infinite loop doesnt happen.
        3. If cell is valid and reachable, mark the visited status as True.
        4. If cell is the final cell then return 1 and mark its visited status as False.
        5. Once all 4 movements are done, mark [r1][c1] visited status to False again.
    '''
    def all_dir_util(m, r1, c1, r, c, visited):
        if r1 > r or r1 < 0 or c1 > c or c1 < 0 or m[r1][c1] == 0:
            return 0
        if visited[r1][c1]:
            return 0
        visited[r1][c1] = True
        if r1 == r and c1 == c:
            visited[r1][c1] = False
            return 1
        result = 0
        
        result += all_dir_util(m, r1-1, c1, r, c, visited)
        result += all_dir_util(m, r1+1, c1, r, c, visited)
        result += all_dir_util(m, r1, c1-1, r, c, visited)
        result += all_dir_util(m, r1, c1+1, r, c, visited)
        visited[r1][c1] = False

        return result

    if m is None or len(m) == 0:
        return 0
    r = len(m) - 1
    c = len(m[0]) - 1
    visited = [[False for x in range(c+1)] for x in range(r+1)]
    return all_dir_util(m, 0, 0, r, c, visited)


if __name__ == "__main__":
    m = [
        [1, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 1]
    ]
    print("Any Path:")
    print(get_any_path(m))
    
    print()
    print("All the Paths:")
    for r in get_all_paths(m):
        print(r)
    
    print()
    print("All the Paths Count:")
    print(get_all_paths_count(m))

    print()
    print("All the Paths Count with all 4 Movements allowed:")
    print(get_all_paths_count_all_dir(m))