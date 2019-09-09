class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1
    def __str__(self):
        return str(self.data)
def get_height(root):
    if root is None:
        return 0
    return max(get_height(root.left), get_height(root.right)) + 1
def create_bst_sorted_array(arr):
    return create_bst_sorted_array2(arr, 0, len(arr) - 1)
def create_bst_sorted_array2(arr, low, high):
    if low > high: return
    mid = (low + high) // 2
    root = Node(arr[mid])
    root.height = max(get_height(root.left), get_height(root.right)) + 1
    root.left = create_bst_sorted_array2(arr, low, mid - 1)
    root.right = create_bst_sorted_array2(arr, mid + 1, high)
    return root

def print_tree(root):
    if root is None: return
    t_matrix = []
    height = get_height(root)
    for r in range(1, height + 1):
        matrix = []
        get_level(root, r, matrix)        
        t_matrix.append(matrix)
    print_matrix(t_matrix)
def get_level(root, level, matrix):
    if root is None: 
        matrix.append(0)
        return
    if level == 1:
        matrix.append(root.data)
    else:
        get_level(root.left, level - 1, matrix)
        get_level(root.right, level - 1, matrix)
def print_matrix(arr):
    data_size = 2
    space = (2**len(arr) - 1) * data_size
    for i in arr:
        l_space = space // len(i)
        for j in i:
            if str(j) == 'X':
                value = 'X'
            else:
                value = str(j)
            if len(i) == 1:
                print(value.center(space + 1, ' '), end = ' ')
            else:
                print(value.center(l_space, ' '), end = ' ')
        print()

def rotate_right(root):
    result = root.left
    root.left = result.right
    result.right = root
    return result
def rotate_left(root):
    result = root.right
    root.right = result.left
    result.left = root
    return result
            
def insert_in_avl(root, key):
    if root is None: return Node(key)
    if root.data < key:
        root.right = insert_in_avl(root.right, key)
    elif root.data > key:
        root.left = insert_in_avl(root.left, key)
    # Balancing Algorithm
    return balance_root(root, key)

def delete_in_avl(root, key):
    if root is None: return root
    if root.data < key:
        root.right = delete_in_avl(root.right, key)
    elif root.data > key:
        root.left = delete_in_avl(root.left, key)
    else:
        if root.left is None:   # 1 or both child are none
            root = root.right
            return root
        elif root.right is None:    # 1 or both child are none
            root = root.left
            return root
        else:
            temp = root.right
            while temp.left:
                temp = temp.left
            root.data = temp.data
            root.right = delete_in_avl(root.right, key)            
    # Balancing Algorithm
    # return balance_root(root, key)
    return root

def balance_root(root, key):
    root.height = get_height(root)
    balance = get_height(root.left) - get_height(root.right)
    if balance > 1 and root.left.data > key:
        return rotate_right(root)
    if balance < -1 and root.right.data < key:
        return rotate_left(root)
    if balance > 1 and root.left.data < key:
        root.left = rotate_left(root.left)
        return rotate_right(root)
    if balance < -1 and root.right.data > key:
        root.right = rotate_right(root.right)
        return rotate_left(root)
    return root

# root = None
# for r in range(1,16):
#     root = insert_in_avl(root, r)
# print_tree(root)
# root = delete_in_avl(root, 1)
# root = delete_in_avl(root, 2)
# root = delete_in_avl(root, 3)
# print_tree(root)

# node = create_bst_sorted_array([x for x in range(1, 32)])
# print_tree(node)
# print(node.height)
