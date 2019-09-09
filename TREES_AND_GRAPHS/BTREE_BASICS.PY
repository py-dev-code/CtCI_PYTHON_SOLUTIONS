from LinkedList import LinkedList 
import random
from collections import Counter

class Node:
    def __init__(self, key):
        self.data = key
        self.left = None
        self.right = None
    def __str__(self):
        return str(self.data)

# Height of a Binary Tree
def tree_height(node):
    if node is None: return 0
    lheight = tree_height(node.left)
    rheight = tree_height(node.right)
    if lheight > rheight:
        return lheight + 1
    else:
        return rheight + 1

def print_level(root, level):
    if root is None: return
    if level == 1:
         print(root, end = ' ')
    else:
        print_level(root.left, level - 1)
        print_level(root.right, level - 1)

def print_tree_bfs(root):
    if root is None: return
    h = tree_height(root)
    for r in range(1, h + 1):
        print_level(root, r)
        print()        

def print_tree_bfs_queue(root):
    if root is None: return
    lst = []
    lst.append(root)
    while len(lst) > 0:
        node = lst.pop(0)
        print(node.data, end = ' ')
        if node.left is not None:
            lst.append(node.left)
        if node.right is not None:
            lst.append(node.right)
        print()

def print_in_order(root):
    if root:
        print_in_order(root.left)
        print(root.data, end = ' ')
        print_in_order(root.right)

def print_pre_order(root):
    if root:
        print(root.data, end = ' ')
        print_pre_order(root.left)        
        print_pre_order(root.right)

def print_post_order(root):
    if root:
        print_post_order(root.left)
        print_post_order(root.right)
        print(root.data, end = ' ')

# work perfectly as long as values are single digits
def print_tree(root):
    h = tree_height(root)
    arr = []
    for r in range(1, h + 1):
        l = []
        get_tree_level(root, r, l)
        arr.append(l)
    # Now, just this array needs to be printed
    size = len(arr)
    space = (2**size - 1)*2
    for i in arr:
        l_space = space // len(i)        
        for j in i:
            if len(i) == 1: print(str(j).center(space + 1, ' '), end = ' ')
            else: print(str(j).center(l_space, ' '), end = ' ')
        print()
            
def get_tree_level(root, level, l):
    if root is None:
        l.append(' ')
        return
    if level == 1:
        l.append(root.data)
    else:
        get_tree_level(root.left, level - 1, l)
        get_tree_level(root.right, level - 1, l)

# Q2: Create a BST with Minimal Height from a sorted array        
def create_bst_sorted_arr(arr):
    root = add_bst_nodes(arr, 0, len(arr) - 1)
    return root
def add_bst_nodes(arr, low, high):
    if low > high: return None
    mid = (low + high) // 2
    node = Node(arr[mid])
    node.left = add_bst_nodes(arr, low, mid - 1)
    node.right = add_bst_nodes(arr, mid + 1, high)
    return node
root = create_bst_sorted_arr([x for x in range(1,16)])
print_tree(root)

# Q3: Create LinkedList for each level of a Binary Tree. For, n levels; we should have n LinkedLists.
def create_ll_list(root):
    h = tree_height(root)
    final_list = []
    for r in range(1, h + 1):
        l = []
        get_tree_level_q2(root, r, l)
        ll = LinkedList(l)
        final_list.append(ll)
    return final_list
def get_tree_level_q2(root, level, l):
    if level == 1 and root is not None: 
        l.append(root.data)
    elif root:
        get_tree_level_q2(root.left, level - 1, l)
        get_tree_level_q2(root.right, level - 1, l)
# node = Node(1)
# node.left = Node(2)
# node.right = Node(3)
# node.left.left = Node(4)
# node.left.right = Node(5)
# node.right.left = Node(6)
# node.right.right = Node(7)
# node.left.left.left = Node(8)
# node.left.left.right = Node(9)
# l = create_ll_list(node)
# for ll in l:
#     print(ll)        

# Q4: Check if a Binary Tree is balanced or not
def get_height_q4(root):
    if root is None: return 0
    return max(get_height_q4(root.left), get_height_q4(root.right)) + 1
def is_balanced(root):
    if root is None: return True
    l1 = get_height_q4(root.left)    
    l2 = get_height_q4(root.right)
    if abs(l1 - l2) > 1: 
        return False
    else:    
        is_balanced(root.left)
        is_balanced(root.right)
    return True
# node = Node(1)
# node.left = Node(2)
# node.right = Node(3)
# node.left.left = Node(4)   
# node.left.left.left = Node(8)
# node.left.left.right = Node(9)
# print(is_balanced(node)) 

# Q5: Check if a Binary Tree is BST or not
# Recursive
def is_BST(root):
    return is_BST_node(root, None, None)
def is_BST_node(root, min, max):
    if root is None: return True
    if (min is not None and root.data < min) or (max is not None and root.data > max):
        return False
    return is_BST_node(root.left, min, root.data) and is_BST_node(root.right, root.data, max)
# In Order Traverse
def is_BST_traverse(root):
    return is_BST_in_order_traverse(root, None)
def is_BST_in_order_traverse(root, last_number):
    if root is None: return True   
    if not is_BST_in_order_traverse(root.left, last_number): return False
    if ((last_number is not None) and (root.data <= last_number)): return False
    last_number = root.data
    if not is_BST_in_order_traverse(root.right, last_number): return False
    return True
# root = create_bst_sorted_arr([1,2,3,4,2,6,7,8])
# print_tree(root)
# print(is_BST(root))
# print(is_BST_traverse(root))

# Q6: Find the successor of a given node in a BST when doing in-order traversal. Each node has its Parent defined as well.
class Node_Q6:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.parent = None
    def __str__(self):
        return str(self.data)
    def add_left_node(self, node):
        self.left = node
        if node:
            node.parent = self
    def add_right_node(self, node):
        self.right = node
        if node:
            node.parent = self
def get_height_q6(root):
    if root is None: return 0
    return max(get_height_q6(root.left), get_height_q6(root.right)) + 1
def create_bst_q6(l):
    return create_bst_parent_q6(l, 0, len(l) - 1)
def create_bst_parent_q6(arr, low, high):
    if low > high: return
    mid = (low + high) // 2
    root = Node_Q6(arr[mid])
    root_left = create_bst_parent_q6(arr, low, mid - 1)
    root_right = create_bst_parent_q6(arr, mid + 1, high)
    root.add_left_node(root_left)
    root.add_right_node(root_right)
    return root
def get_successor_q6(node):
    if node is None: return None
    if node.right:
        n = node.right
        while n:
            if n.left: 
                n = n.left
            else:
                return n
    else:
        n = node
        x = n.parent        
        while x is not None:
            if x.left == n: return x
            n = x
            x = x.parent
# node = create_bst_q6([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
# print_tree(node)
# print(node.left.right.right.right.data)
# print(get_successor_q6(node.left.right.right.right).data)

# Q8: First Common Ancestor of 2 given nodes in a Tree [not necessarily a Binary Tree]

# First Solution is with Assumption that each node has a Link to its parent. This will be done via Depth calculation.
def get_comm_anc_parent_link_q8_depth_approach(node1, node2): # root node will not be needed in this method.
    d1 = get_depth(node1)
    d2 = get_depth(node2)
    if d1 > d2:
        deep = node1
        shallow = node2
    else:
        deep = node2
        shallow = node1
    for _ in range(abs(d1 - d2)):
        deep = deep.parent
    while deep:
        if deep.parent == shallow.parent: 
            return deep.parent
        else:
            deep = deep.parent
            shallow = shallow.parent
def get_depth(node):
    depth = 0
    while node.parent:
        depth += 1
        node = node.parent
    return depth
# node = create_bst_q6([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
# print_tree(node)
# n1 = node.left.right.left
# n2 = node.left.right.right.right
# print('{} - {}'.format(n1.data, n2.data)) 
# result = get_comm_anc_parent_link_q8_depth_approach(n1, n2)
# print(result.data)

# 2nd Solution: Assumption of Parent being present and traversing upwords until subtree of the parent is covering other node.
def get_comm_anc_parent_link_upward_travers_approach_q8(root, node1, node2):
    if root is None: return 'Root is None'
    if does_cover(root, node1) == False or does_cover(root, node2) == False:
        return 'At least one node is not present in the Tree'
    elif does_cover(node1, node2):
        return node1
    elif does_cover(node2, node1):
        return node2
    worker = node1
    parent = worker.parent
    while parent:
        if parent.left == worker:
            sibling = parent.right
        else:
            sibling = parent.left
        if does_cover(sibling, node2): 
            break
        worker = parent
        parent = parent.parent
    return parent    
def does_cover(root, node):
    if root is None: return False
    if root == node: return True
    return does_cover(root.left, node) or does_cover(root.right, node)
# node = create_bst_q6([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
# n1 = node.left.left.left
# n2 = node.left.right.right.right
# print_tree(node)    
# print(n1, n2)
# result = get_comm_anc_parent_link_upward_travers_approach_q8(node, n1, n2)
# print(result)
    
# 3rd Solution: When no link to Parent is given. We will start with root node to solve the problem.
def get_comm_anc_no_parent_q8(root, node1, node2):
    if root is None: return None
    if (not does_cover1(root, node1)) or (not does_cover(root, node2)): return False        
    return get_comm_anc_no_parent_q8_helper(root, node1, node2)
def get_comm_anc_no_parent_q8_helper(root, node1, node2):    
    if ((root is None) or (root == node1) or (root == node2)): return root
    is_1_left = does_cover1(root.left, node1)
    is_2_left = does_cover1(root.left, node2)
    if is_1_left != is_2_left:
        return root
    else:
        if is_1_left:
            return get_comm_anc_no_parent_q8_helper(root.left, node1, node2) 
        else:
            return get_comm_anc_no_parent_q8_helper(root.right, node1, node2)
def does_cover1(root, node):
    if root is None: return False
    if root == node: return root
    return does_cover1(root.left, node) or does_cover1(root.right, node)
# node = create_bst_q6([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
# n1 = node.right.left.left
# n2 = node.right.right.right.right
# print_tree(node)    
# print(n1, n2)
# result = get_comm_anc_parent_link_upward_travers_approach_q8(node, n1, n2)
# print(result)   

# Q10: Given 2 Binary Trees T1 and T2. Find out if T2 is a subtree of T1 or not. 
# [Only value match needs to be done as nodes of both the trees can be different.] 
# Throw some light on the optimization when T1 is really big and T2 is a small one.

# Solution 1: Cost: O(n + m): Make a string of both the trees by doing a Pre-Order Traverse and put X where node is null.
def is_subtree_pre_order_traverse_q10(node1, node2):
    if node1 is None or node2 is None: return False
    list1 = []
    list2 = []
    make_pre_order_string(node1, list1)
    make_pre_order_string(node2, list2)
    if ''.join(list2) in ''.join(list1): 
        return True
    else:
        return False
def make_pre_order_string(node, result):
    if node is None: 
        result.append('X')
        return
    result.append(str(node.data))
    make_pre_order_string(node.left, result)
    make_pre_order_string(node.right, result)
# node1 = create_bst_sorted_arr([1,2,3,4,5,6,7])
# node2 = create_bst_sorted_arr([5])
# print_tree(node1)
# print()
# print_tree(node2)
# print(is_subtree_pre_order_traverse_q10(node1, node2))

# Solution2: We will do pre-order traverse of T1 [bigger tree] 
# and if we find a node that matches T2 then we will start comparison of T1 and T2.
def is_subtree_approach2_q10(node1, node2):
    if node2 is None: return True # Blank node is always a Subtree
    return subtree(node1, node2)
def subtree(node1, node2):
    if node1 is None: 
        return False
    elif node1.data == node2.data and match_tree(node1, node2):
        return True
    else:
        print('in')
        return subtree(node1.left, node2) or subtree(node1.right, node2)
def match_tree(node1, node2):
    if node1 is None and node2 is None:
        return True
    elif node1 is None or node2 is None:
        return False
    elif node1.data != node2.data:
        return False
    else:
        return match_tree(node1.left, node2.left) and match_tree(node1.right, node2.right)
# node1 = create_bst_sorted_arr([1,2,3,4,5,6,7])
# node2 = create_bst_sorted_arr([5,6,7])
# print_tree(node1)
# print()
# print_tree(node2)
# print(is_subtree_approach2_q10(node1, node2))     

# Q11: Random Node: Design a Binary Tree with insert, find, delete and get_random_node() methods 
# where all the nodes has same probability to be selected in get_random method.   
# Approach: We will do this by maintaining a size attribute for each node and its value will tell the number of nodes below this node.
# We will maintain Binary Tree in a BST Fashion
class Node_Q11:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.size = 1
    def __str__(self):
        return str(self.data)
    def insert_node_in_order(self, d):
        if d <= self.data:
            if self.left is None: self.left = Node_Q11(d)
            else: self.left.insert_node_in_order(d)
        else:
            if self.right is None: self.right = Node_Q11(d)
            else: self.right.insert_node_in_order(d)
        self.size += 1
    def find(self, d):
        if self.data == d:
            return self
        elif self.data > d and self.left:
            return self.left.find(d)
        elif self.data < d and self.right:    
            return self.right.find(d)
        else:
            return None
    def get_random_node1(self): # In this we will pick a random number each time and decide where to go to pick the node.
        if self.left:
            left_size = self.left.size
        else:
            left_size = 0
        rand = random.randint(0, self.size - 1)
        if rand < left_size:
            return self.left.get_random_node1()
        elif rand == left_size:
            return self
        else:
            return self.right.get_random_node1()
    def get_random_node2(self): # In this we will pick a random number and traverse the tree to return the node at the same number.
        rand = random.randint(0, self.size - 1)
        return self.get_ith_node(rand)
    def get_ith_node(self, i):
        if self.left:
            left_size = self.left.size
        else:
            left_size = 0
        if i < left_size:
            return self.left.get_ith_node(i)
        if i == left_size:
            return self
        else:
            return self.right.get_ith_node(i - (left_size + 1))    
    def delete_last_node(self):
        # will implement this while learning algorithms for self-balancing trees
        pass        
# node = Node_Q11(4)
# node.insert_node_in_order(2)
# node.insert_node_in_order(1)
# node.insert_node_in_order(3)
# node.insert_node_in_order(6)
# node.insert_node_in_order(5)
# node.insert_node_in_order(7)
# node.insert_node_in_order(8)
# # print(node.find(8))
# print_tree(node)
# l = []
# for _ in range(5000):
#     n = node.get_random_node2()
#     l.append(n.data)
# c = Counter(l)
# l2 = []
# for k in c.keys():
#     l2.append(k)
# l2 = [x for x in sorted(l2)]
# for k in l2:
#     print('{}-{};'.format(k, c[k]), end = ' ')

# Q12: Return the number of paths where sum of nodes equal to a given value. Path can only go downwards.
def get_number_of_paths(node, target_sum):
    if node is None: return 0
    root_count = get_path_from_node(node, target_sum, 0)
    root_count += get_number_of_paths(node.left, target_sum)
    root_count += get_number_of_paths(node.right, target_sum)
    return root_count
def get_path_from_node(node, target_sum, cum_sum):
    if node is None: return 0
    total_paths = 0
    cum_sum += node.data
    if cum_sum == target_sum: total_paths += 1
    total_paths += get_path_from_node(node.left, target_sum, cum_sum)
    total_paths += get_path_from_node(node.right, target_sum, cum_sum)
    return total_paths
# node = create_bst_sorted_arr([1,1,1,1,1,2,2,2])     
# print_tree(node)   
# print(get_number_of_paths(node, 2))

# Approach 2: Optimal Solution where we keep track of the RunningSum and get it done via a HashMap or Dictionary
# The solution below has a bug as its output is not correct
# def get_number_of_paths2(node, target_sum):
#     if node is None: return 0
#     d = {}
#     return count_paths_with_sum(node, target_sum, 0, d)
# def count_paths_with_sum(node, target_sum, running_sum, d):
#     if node is None: return 0
#     running_sum += node.data
#     sum = running_sum - target_sum
#     try:
#         total_paths = d[sum]
#     except:
#         total_paths = 0
#     if running_sum == target_sum: 
#         total_paths += 1
#     # increment the values of the key "sum"
#     try:
#         val = d[sum]
#         d[sum] = val + 1
#     except:
#         d[sum] = 1
#     total_paths += count_paths_with_sum(node.left, target_sum, running_sum, d)  
#     total_paths += count_paths_with_sum(node.right, target_sum, running_sum, d)
#     # decrement the values of the key "sum"
#     try:
#         val = d[sum]
#         d[sum] = val - 1
#     except:
#         d[sum] = 0
#     print(d)
#     return total_paths
# node = create_bst_sorted_arr([1,1,1,1,1,2,2,2])     
# print_tree(node)   
# print(get_number_of_paths2(node, 2))    

def create_tree_With_list(l):
    pass


    

