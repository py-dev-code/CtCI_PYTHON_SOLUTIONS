import BTREE_BASICS as bt

# Searching a key in BST
def search(root, key):
    if root is None or root.data == key:
        return root
    elif root.data < key:
        return search(root.right, key)
    else:
        return search(root.left, key)    
# node = bt.create_bst_sorted_arr([1,2,4,5,6,7,8])
# bt.print_tree(node)
# n = search(node, 6)
# print(n)

# Inserting into a BST
def insert(root, key):
    if root is None: 
        root = bt.Node(key)
    else:
        if root.data < key:
            if root.right is None:
                root.right = bt.Node(key)
            else:
                insert(root.right, key)
        else:
            if root.left is None:
                root.left = bt.Node(key)
            else:
                insert(root.left, key)
# node = bt.create_bst_sorted_arr([1,2,4,5,6])
# bt.print_tree(node)
# insert(node, 3)
# bt.print_tree(node)

# Delete a key from BST
def delete(root, key):
    if root is None:
        return root
    if root.data > key:
        root.left = delete(root.left, key)
    elif root.data < key:
        root.right = delete(root.right, key)
    else:
        # only 1 child is None or both child are None
        if root.left is None:
            temp = root.right
            root = None
            return temp
        elif root.right is None:
            temp = root.left
            root = None
            return temp
        # Both child are not None then next successor should be returned
        temp = root.right
        while temp.left:
            temp = temp.left
        root.data = temp.data
        root.right = delete(root.right, temp.data)
    return root
# node = bt.create_bst_sorted_arr([1,2,4,5,6,7,8,9,10])
# bt.print_tree(node)
# delete(node, 6)
# bt.print_tree(node)


