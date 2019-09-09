def zeroMatrix(matrix):

    m = len(matrix)
    n = len(matrix[0])
    rows = []
    cols = []

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                rows.append(i)
                cols.append(j)

    for row in rows:
        nullify_row(matrix, row)

    for col in cols:
        nullify_col(matrix, col)

def nullify_row(matrix, row):
    for i in matrix[row]:
        matrix[row][i] = 0

def nullify_col(matrix, col):
    for i in range(len(matrix)):
        matrix[i][col] = 0

lst = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 1]
        ]

zeroMatrix(lst)        
print(lst)