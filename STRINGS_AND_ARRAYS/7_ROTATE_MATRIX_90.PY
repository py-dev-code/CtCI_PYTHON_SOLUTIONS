def rotate_matrix(matrix):
    n = len(matrix)

    for layer in range(n // 2):
        first = layer
        last = n - layer - 1
        for i in range(first, last):
            # save top to a variable
            top = matrix[layer][i]

            # left to top
            matrix[layer][i] = matrix[- i - 1][layer]

            # bottom to left
            matrix[- i - 1][layer] = matrix[- layer - 1][- i - 1]

            # right to bottom
            matrix[- layer - 1][- i - 1] = matrix[i][- layer - 1]

            # top to right
            matrix[i][- layer - 1] = top

    return matrix        

matrix = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]    

print(rotate_matrix(matrix))