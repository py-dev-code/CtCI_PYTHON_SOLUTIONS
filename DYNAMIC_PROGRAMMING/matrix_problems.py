def get_total_paths_normal(matrix):
    '''
    Condition: Only right and down movement is allowed. Cell with 0 value are not reachable.
    '''    
    def get_total_paths_util_normal(matrix, r, c):
        if r == row-1 and c == col-1:
            return 1
        path_count = 0
        # Right cell
        x_right = r
        y_right = c+1
        if x_right < row and y_right < col and matrix[x_right][y_right] == 1:
            path_count += get_total_paths_util_normal(matrix, x_right, y_right)
        # Down cell
        x_down = r+1
        y_down = c
        if x_down < row and y_down < col and matrix[x_down][y_down] == 1:
            path_count += get_total_paths_util_normal(matrix, x_down, y_down)
        return path_count

    row = len(matrix)
    col = len(matrix[0])
    if matrix[0][0] == 0 or matrix[row-1][col-1] == 0:
        return 0
    return get_total_paths_util_normal(matrix, 0, 0)


def get_total_paths_memo(matrix):
    '''
    Condition: Only right and down movement is allowed. Cell with 0 value are not reachable.
    '''      
    def get_total_paths_util_memo(matrix, r, c):
        if count_matrix[r][c] is not None:
            # print('Giddy up')
            return count_matrix[r][c]
        if r == row-1 and c == col-1:
            return 1
        path_count = 0
        # Right cell
        x_right = r
        y_right = c+1
        if x_right < row and y_right < col and matrix[x_right][y_right] == 1:
            path_count += get_total_paths_util_memo(matrix, x_right, y_right)
        # Down cell
        x_down = r+1
        y_down = c
        if x_down < row and y_down < col and matrix[x_down][y_down] == 1:
            path_count += get_total_paths_util_memo(matrix, x_down, y_down)
        count_matrix[r][c] = path_count
        return path_count

    row = len(matrix)
    col = len(matrix[0])
    if matrix[0][0] == 0 or matrix[row-1][col-1] == 0:
        return 0
    count_matrix = [[None for x in range(col)] for x in range(row)]
    return get_total_paths_util_memo(matrix, 0, 0)


def get_all_paths(matrix):
    '''
    Condition: Only right and down movement is allowed. Cell with 0 value are not reachable.
    '''
    def get_all_paths_util(matrix, r, c, path, paths):
        point = (r, c)
        path.append(point)
        if r == row-1 and c == col-1:
            paths.append(list(path))
        else:
            # Right cell
            x_right = r
            y_right = c+1
            if x_right < row and y_right < col and matrix[x_right][y_right] == 1:
                get_all_paths_util(matrix, x_right, y_right, path, paths)
            # Down cell
            x_down = r+1
            y_down = c
            if x_down < row and y_down < col and matrix[x_down][y_down] == 1:
                get_all_paths_util(matrix, x_down, y_down, path, paths)
        path.pop()

    row = len(matrix)
    col = len(matrix[0])
    if matrix[0][0] == 0 or matrix[row-1][col-1] == 0:
        return 0
    path = []
    paths = []
    get_all_paths_util(matrix, 0, 0, path, paths)
    return paths

def count_paths(N, prison):
    '''
    Total possible paths when all 4 movements are allowed. Cell with value 0 are not reachable.
    Below is the pseudo code.
    Its exactly same as right-down approach except here, we need to maintain a matrix for visited cells to avoid
    infinite loops.
    '''

    # def count_paths_recurse(X, Y, prison, visited):
    #     if X == N-1 and Y == N-1:  # reached the exit cell
    #         return 1
    #     visited[X][Y] = True
    #     pathcount = 0
    #     for direction in (right, left, down, up):
    #         Xnew, Ynew = neighoring cell coordinates in that direction
    #         if Xnew and Ynew are inside the prison
    #                 and prison[Xnew][Ynew] == 0
    #                 and not visited[Xnew][Ynew]:
    #             pathcount += count_paths_recurse(Xnew, Ynew, prison, visited)
    #     visited[X][Y] = False
    #     return pathcount

    # if prison is not an NxN matrix containing only 0s and 1s:
    #     raise an error
    # create visited as an NxN matrix containing only False
    # if prison[0][0] != 0 or prison[N-1][N-1] != 0:
    #     return 0
    # return count_paths_recurse(0, 0, prison, visited)    


if __name__ == "__main__":
    matrix = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    print(get_total_paths_memo(matrix))
    print(get_total_paths_normal(matrix))
    all_paths = get_all_paths(matrix)
    for r in all_paths:
        print(r)
