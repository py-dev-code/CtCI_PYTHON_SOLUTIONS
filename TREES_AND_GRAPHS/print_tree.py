def print_tree(root, height, max_node_len):
    if root is None: return
    l = [None, root]
    for level in range(2, height + 1):
        process_level(root, level, l)
    max_row_len = 2**(height - 1) * max_node_len * 2
    for r in range(1, height + 1):
        print_level(r, l, max_row_len)
        print()
    
def print_level(level, l, max_row_len):
    if level == 1:
        num_of_nodes = 1
    else:
        num_of_nodes = 2**(level - 1)
    node_space = max_row_len // num_of_nodes
    for r in get_level_index(level):
        if l[r] == 'x':
            print(' '.center(node_space, ' '), end = '')
        else:
            print(str(l[r]).center(node_space, ' '), end = '')
    
def get_level_index(level):
    return list(range(2**(level-1), 2**(level)))

def process_level(root, level, l):
    for r in get_level_index(level - 1):
        if l[r] == 'x':
            l.append('x')
            l.append('x')
        else:
            if l[r].left is None:
                l.append('x')
            else:
                l.append(l[r].left)
            if l[r].right is None:
                l.append('x')
            else:
                l.append(l[r].right)
