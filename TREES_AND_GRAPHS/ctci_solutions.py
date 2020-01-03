import random

class ListNode(object):
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
    def __repr__(self):
        return str(self.data)

class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None
    def add_in_start(self, value):
        if self.head is None:
            self.head = self.tail = ListNode(value)
        else:
            self.head = ListNode(value, self.head)
    def add_in_end(self, value):
        if self.head is None:
            self.head = self.tail = ListNode(value)
        else:
            self.tail.next = ListNode(value)
            self.tail = self.tail.next
    def __repr__(self):
        result = []
        node = self.head
        while node:
            result.append(str(node.data))
            node = node.next
        return ' -> '.join(result)    

class Graph(object):    
    def __init__(self, graph_dict=None):
        self.graph_dict = graph_dict
    def add_vertex(self, vertex):
        try:
            _ = self.graph_dict[vertex]
        except KeyError:
            self.graph_dict[vertex] = []
    def add_edge(self, v1, v2):
        self.add_vertex(v1)
        self.add_vertex(v2)
        self.graph_dict[v1].append(v2)
    def __repr__(self):
        result = '{\n'
        for r in self.graph_dict:
            result += str(r) + ':'
            result += str(self.graph_dict[r]) + "\n"
        result += "}"
        return result

class Node(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right    
    def __repr__(self):
        return str(self.data)

class BTree(object):    
    def __init__(self, root=None):
        self.root = root
    def get_height(self):
        def get_height_util(root):
            if root is None:
                return 0
            return max(get_height_util(root.left), get_height_util(root.right)) + 1
        return get_height_util(self.root)  
    def print(self):
        print_tree(self.root)  
    
def print_tree(root, max_node_len=2):
    def get_root_height(root):
        if root is None: return 0
        return max(get_root_height(root.left), get_root_height(root.right)) + 1
    def get_parents(level, nodes):
        return nodes[2**level - 1: 2**(level+1) - 1]
    def process_level(root, level, nodes):
        node_length = line_length // (2**level)
        if level == 0:
            nodes.append(root)
            print_value = str(root.data)
            print(print_value.center(node_length, ' '), end='')
        else:
            for r in get_parents(level - 1, nodes):
                value = 'X' if r == 'X' or r.left is None else r.left
                print_value = '' if value == 'X' else str(value.data)
                nodes.append(value)
                print(print_value.center(node_length, ' '), end='')

                value = 'X' if r == 'X' or r.right is None else r.right
                print_value = '' if value == 'X' else str(value.data)
                nodes.append(value)
                print(print_value.center(node_length, ' '), end='')        
    if root is None: return
    height = get_root_height(root)
    line_length = 2**(height-1) * 2 * max_node_len
    nodes = []
    for level in range(height):
        process_level(root, level, nodes)
        print()        


# Problem: 4.1 Route Between Nodes: Given a directed graph, design an algorithm to find out whether there is a
# route between two nodes.
def route_between_nodes(graph, src, dest):
    '''
    Algorithm:
        This can be done as a straight forward BFS or DFS. Below is an iterative BFS implementation with list being used
        as a Queue.
    '''
    def route_between_nodes_util(graph, src, dest, visit_status, queue):
        visit_status[src] = True
        queue.append(src)
        while len(queue) > 0:
            node = queue.pop(0)
            if node == dest:
                return True
            for nbhr in graph.graph_dict[node]:
                if not visit_status[nbhr]:
                    queue.append(nbhr)
                    visit_status[nbhr] = True
        return False

    visit_status = {}
    for r in graph.graph_dict:
        visit_status[r] = False
    queue = []
    return route_between_nodes_util(graph, src, dest, visit_status, queue)    


# Problem:4.2 Minimal Tree: Given a sorted (increasing order) array with unique integer elements, write an algorithm
# to create a binary search tree with minimal height.
def create_min_height_bst(array):
    '''
    Algorithm:
        The Trick is to start the root with mid element of the array.
        Left node will be the middle of left portion of the array and Right node will be the middle or right portion.
    '''
    def create_min_height_bst_util(array, low, high):
        if low > high:
            return
        mid = (low + high) // 2
        node = Node(array[mid])
        node.left = create_min_height_bst_util(array, low, mid - 1)
        node.right = create_min_height_bst_util(array, mid + 1, high)
        return node        
    
    if array is None or len(array) == 0:
        return
    tree = BTree()    
    tree.root = create_min_height_bst_util(array, 0, len(array) - 1)
    return tree


# Problem:4.3 List of Depths: Given a binary tree, design an algorithm which creates a linked list of all the nodes
# at each depth (e.g., if you have a tree with depth D, you'll have D linked lists).
def list_of_depths(root):
    '''
    Algorithm:
        First get the height of the root.
        Then call a function for each height level which will append all the nodes in the list for the given level.
        Function will append the node in the list if level is 1 else it will recurse to left and right node with level - 1.
    '''
    def get_height(root):
        if root is None: return 0
        return max(get_height(root.left), get_height(root.right)) + 1
    
    def process_level(root, level, ll):
        if root is None: return
        if level == 1:
            ll.add_in_end(root.data)
        else:
            process_level(root.left, level - 1, ll)
            process_level(root.right, level - 1, ll)

    if root is None: return []
    for level in range(1, get_height(root) + 1):
        ll = LinkedList()
        process_level(root, level, ll)
        print(ll)


# Problem:4.4 Check Balanced: Implement a function to check if a binary tree is balanced. For the purposes of
# this question, a balanced tree is defined to be a tree such that the heights of the two subtrees of any
# node never differ by more than one.
def is_balanced(root):
    '''
    Algorithm:
        If None root is given then we will return True. 
        Get the height of left and right subtree, if difference is more than 1, return False.
        Recurse the method for left and right subtree and return the result by putting an "and" between the two.
    '''    
    def get_height_util(root):
        if root is None:
            return 0
        return max(get_height_util(root.left), get_height_util(root.right)) + 1
    
    if root is None:
        return True
    
    lheight = get_height_util(root.left)
    rheight = get_height_util(root.right)

    if abs(lheight - rheight) > 1:
        return False
    else:
        return is_balanced(root.left) and is_balanced(root.right)


# Problem:4.5 Validate BST: Implement a function to check if a binary tree is a binary search tree.
def validate_bst(root):
    '''
    Algorithm:
        To validate if a Tree is a BST, we need to check the following condition:
            Each Node value is more than or equal to all the nodes in its Left Subtree and 
            Each Node value is less than all the nodes in its Right Subtree.
    Implementation#1:
        1. We will implement this method recursively by passing min and max values.
        2. Initially, min and max will be set to None.
        3. If a node's value is more than max (it max is not None) or less than or equal to min (if min is not None) then
        return False.
        4. While moving left, we will update the max value with node's value and while moving right, we will
        update the min value with node's value.
    Another solution for this will be to do a In-order Traversal of the Tree and create a list from it. If that list is
    sorted then its a BST else not. But this approach will not work when Tree has duplicate nodes becuase list wont know the 
    exact position of the nodes. In below 2 trees, both the list will look like 20-20 but 2nd Tree is not a BST while 1st Tree is.
        20          20
    20                  20
    '''
    def validate_bst_util(root, min, max):
        if root is None:
            return True
        if (min is not None and root.data <= min) or (max is not None and root.data > max):
            return False
        return validate_bst_util(root.left, min, root.data) and validate_bst_util(root.right, root.data, max)

    return validate_bst_util(root, None, None)


# Problem:4.6 Successor: Write an algorithm to find the "next" node (i.e., in-order successor) of a given node in a
# binary search tree. You may assume that each node has a link to its parent.
def get_successor():
    '''
    Since Normal Binary Tree does not store the Parent Node links so we will define a custom Node for this function.
    We will create a Binary Search Tree inside this function only and implement the algorithm within a util function.
    Algorithm:
        Successor can be determined by 1 of the 2 secnarios:
        1. If node.right is not None then successor will be the left most node of the right subtree.
        2. If 1 is not the case then we need to go up in the Hierarchy and find the ancestor whose left node is the node 
        currently active in the loop. We will take parent node in p and given node in n.
        Run a while loop till p is not None. If p.left is n then return p else p will become p.parent and n will become p.
        If we reach the root and no match is found then no successor is there for the given 
        node.
    '''

    class PNode(object):
        def __init__(self, data, parent=None, left=None, right=None):
            self.data = data
            self.parent = parent
            self.left = left
            self.right = right
        def __repr__(self):
            return str(self.data)
    
    def get_successor_util(node):
        if node is None:
            return
        if node.right:
            n = node.right
            while n.left:
                n = n.left
            return n
        else:
            p = node.parent
            n = node
            while p:
                if p.left == n:
                    return p
                else:
                    n = p
                    p = p.parent
            return None

    # Creating the Tree
    root = PNode(5)
    root.left = PNode(2, root)
    root.right = PNode(7, root)
    root.left.left = PNode(1, root.left)
    root.left.right = PNode(3, root.left)
    root.right.left = PNode(6, root.right)
    root.right.right = PNode(8, root.right)
    root.left.right.right = PNode(4, root.left.right)
    root.right.right.right = PNode(9, root.right.right)

    # Printing the Tree for visualizing the Tree
    print('Tree of the problem is: ')
    print_tree(root, 2)

    node = root.right.right.right
    result = get_successor_util(node)    
    print(f'\nSuccessor of Node {node} is: {result}')


# Problem:4.7 Build Order: You are given a list of projects and a list of dependencies (which is a list of pairs of
# projects, where the second project is dependent on the first project). All of a project's dependencies
# must be built before the project is. Find a build order that will allow the projects to be built. If there
# is no valid build order, return an error.
# EXAMPLE
# Input:
# projects: a, b, c, d, e, f
# dependencies: (a, d), (f, b), (b, d), (f, a), (d, c)
# Output: f, e, a, b, d, c
def graph_topological_sort(graph):
    '''
    Algorithm:
        This problem is an example of Topological Sorting.
        How it works:
            Find nodes that do not have any incoming edges. Once we find them, we will remove all the outgoing edges 
            from them. If we can not find any such nodes then Build Order or Topological Sorting is not possible.
            Recurse the above step for remaining Graph structure.
        Implementation:
            We will implement the Graph as an adjacency list here.
            We will define a function that will take graph dictionary and an empty list. This function will be recursively called.
            If length of the dictionary is 0 return False.
            Run a loop for dict keys. Run another loop for all dict values which are a list. If key is found in any of the values 
            then this key will not be deleted else Delete the dict key and append the same key in result list.
            After the loop, check if we have deleted any keys or not. If no key is deleted then return False.
            We cannot delete key from dictionary while looping through it so we can use a variable to do all deletes after the 
            loop.
    '''    
    def topological_sort_util(d_util, result):
        if len(d_util) == 0:
            return True               
        for key in d_util:
            del_key = None
            delete = True
            for nodes in d_util.values():
                if key in nodes:
                    delete = False
                    break
            if delete: 
                del_key = key
                break
        if del_key is None: return False
        del d_util[del_key]
        result.append(del_key)
        return topological_sort_util(d_util, result)

    if graph is None or graph.graph_dict is None: return    
    d_util = dict(graph.graph_dict)
    result = []
    if topological_sort_util(d_util, result):
        return result
    else:
        return "Build Order Not Possible."


# Problem:4.8 First Common Ancestor: Design an algorithm and write code to find the first common ancestor
# of two nodes in a binary tree. Avoid storing additional nodes in a data structure. NOTE: This is not
# necessarily a binary search tree.
def common_ancestor(root, node1, node2):
    '''
    Algorithm:
        Basic algorithm here is to keep checking whether both given nodes belong to different side of the root or not. If yes, root is the ancestor else recurse the process to the side where both node belongs.
    Implementation:
        Create a util function that can return whether a given node belnogs to the root or not.
        If root is same as node1 or node2 then return root as common ancestor will be the matching node only.
        We will take root.left and check whether it covers node1 and node2. If answer is same for both the calls then it 
        means that we will recurse the call for either root.left (when both answers are True) or root.right (when both 
        answers are False). If answer is different then return the root.
    '''
    def covers(root, node):
        if root is None:
            return False
        if root.data == node:
            return True
        return covers(root.left, node) or covers(root.right, node)

    def common_ancestor_util(root, node1, node2):
        if root is None or root.data == node1 or root.data == node2:
            return root
        is_left1 = covers(root.left, node1)
        is_left2 = covers(root.left, node2)
        if is_left1 != is_left2:
            return root
        subtree_root = root.left if is_left1 else root.right
        return common_ancestor_util(subtree_root, node1, node2)

    if not covers(root, node1) or not covers(root, node2): return None
    return common_ancestor_util(root, node1, node2)


# Problem:4.9 BST Sequences: A binary search tree was created by traversing through an array from left to right
# and inserting each element. Given a binary search tree with distinct elements, print all possible
# arrays that could have led to this tree.
# EXAMPLE
# Input: Node(2, Node(1, None, None), Node(3, None, None))
# Output: {2, 1, 3}, {2, 3, 1}
def all_bst_array_sequences(root):
    '''
        Algorithm:
            This is a compliacated algorithm. Lets say we have a BST of 3 nodes with root as 2, left as 1 and right as 3.
            This Tree can be created with 2 arrays: [2,1,3] and [2,3,1].
            This means that if we loop through this array and insert each element in a BST then final tree will be our 
            original Tree.
            Observation 1: First element should always be 2 as it needs to be root. Once root is done, doesnt matter what 
            comes next as BST insert algorithm will route all nodes smaller than 2 to the left of root and so on for nodes 
            more than root.
            Observation 2: We can switch left and right subtree to make all the combinations. This combination process needs
            to be analysed in more detail.
                        2.5
                    2       3
                1               4
            Above BST is our example Tree.
            Observation 3: If we know all the possible arrays that can result into left subtree and all possible
            arrays of right subtree then we know by combining the left and right list togather and if we append root 
            to each of that list; we will have our final list of arrays.
            Weaving Algorithm:
                In our example, we can say that left subtree has a possible array of [2,1] and right subtree has an array of
                [3,4]. We need to combine these 2 arrays now, how? 
                Weaving of array algorithm will combine these 2 arrays and give the list of all the arrays.
                So, we can try all the possible scenarios of both arrays as long as sequence of individual array is matched.
                Means, in all arrays 2 should always come before 1 and 3 should always come before 4.
            Waeving Algorithm Implementation:
                Input will be 3 lists: left array [2,1], right array [3,4] and prefix [2.5]
                How it will recurse:
                    We will take out left array and pop its 1st element and append it to the prefix.
                    So next recursion will be [1], [3,4] and [2.5, 2]
                    Repeating the above process and next recursion will be: [], [3,4] and [2.5, 2, 1]
                    And we got to our base case when one of the list is empty.
                    we will create a list [2.5, 2, 1, 3, 4] and add it to the final result set.
                    2 things: we will need to access the same list for each fresh recursion so we will make a copy of 
                    left and right list to do the recursion.
                    same applies to prefix list as well but this can be done in place only. Once the left list is done, 
                    we will pop the last element out from the prefix to make it ready.
                    Once left array is done, same thing will be repeated for the right list.
            Recursion Algorithm for calling Weaving method:
                We need to weave left and right arrays in all possible combinations as long as order is maintained.
                So this will be done by 2 loops:
                    Weaving list output will be a list of list.
                    So, once weaving for left and right subtree is done we will have 2 2-D lists.
                        for left in left_list:
                            for right in right_list:
                                weaving_call(left, right, prefix=[root.data])
                    Above looping mechanism will call the weaving for any particular node.
                    Base case: If node is None then this method will return a list which has a list with 0 elements.
                    Else: prefix will be created by adding root data. Then we will call the same method for left and right subtree
                    to get the left and right list.
            Below code implements the same approach.
    '''    
    def weave_lists(left, right, weaved, prefix):
        if len(left) == 0 or len(right) == 0:
            result = list(prefix)
            result.extend(left)
            result.extend(right)
            weaved.append(result)
            return
        
        left_copy = list(left)
        right_copy = list(right)
        
        left_first = left_copy.pop(0)
        prefix.append(left_first)
        weave_lists(left_copy, right, weaved, prefix)
        prefix.pop()

        right_first = right_copy.pop(0)
        prefix.append(right_first)
        weave_lists(left, right_copy, weaved, prefix)
        prefix.pop()

    results = []
    if root is None:
        results.append([])
        return results

    prefix = [root.data]
    left_seq = all_bst_array_sequences(root.left)
    right_seq = all_bst_array_sequences(root.right)
    
    for left in left_seq:
        for right in right_seq:
            weaved = []
            weave_lists(left, right, weaved, prefix)
            for r in weaved:
                results.append(r)
    
    return results


# Problem:4.10 Check Subtree: T1 and T2 are two very large binary trees, with T1 much bigger than T2. Create an
# algorithm to determine if T2 is a subtree of T1.
# A tree T2 is a subtree of T1 if there exists a node n in T1 such that the subtree of n is identical to T2.
# That is, if you cut off the tree at node n, the two trees would be identical.
def check_subtree1(root1, root2):
    '''
    Algorithm:
        We can find out if root2 is a subtree of root1 by traversing both the trees but which Traversal needs to be done.
        Since we are comparing the Trees starting from their root, we need to do in-order Traversal.
        Now, a normal in-order traversal can give false results when nodes are None so in order to handle None nodes, we will
        append 'X' in the final in-order output string. This way, each Tree in-order output string will be unique.
        If tree2 output is a substring of tree1 then return True else return False.
        Time Complexity of this algorithm will be O(n+m) where tree1 has n nodes and tree2 has m nodes.
    '''
    def in_order_util_string(root, array):
        if root is None: 
            array.append('X')
            return
        array.append(str(root.data))
        in_order_util_string(root.left, array)
        in_order_util_string(root.right, array)

    if root1 is None or root2 is None:
        return False
    result1 = []
    in_order_util_string(root1, result1)
    result2 = []
    in_order_util_string(root2, result2)
    print(''.join(result1))
    print(''.join(result2))
    return ''.join(result2) in ''.join(result1)

def check_subtree2(root1, root2):
    '''
    Algorithm:
        In this algorithm, we traverse (in-order) through bigger tree and compare each of its node to the root of tree2.
        If they match, we call a util method match_tree that will check both the given roots have same data or not.
        If they do not match, we continue to left and right of tree1.
        Time complexity of this algorithm will be O(n + km) where n is number of nodes in Tree1, m is number of nodes in Tree2
        and k is number of nodes in Tree1 that has same value as Tree2 root. So, worst case complexity will be O(nm).
        If we simulate both the trees with Random numbers then value of k will be n/p where p is the integer in which all the root
        values are present. By putting, n = 10000, m = 100 and p = 1000; an average complexity will be 
        10000 + 10000/1000 * 100 = 11000. 
        This gives us a base to compare both the algorithms and trading between a better average time (algorithm 2) or a better worst case time (algorithm 1).
    '''
    def match_tree(root1, root2):
        if root1 is None and root2 is None:
            return True
        elif root1 is None or root2 is None:
            return False
        elif root1.data != root2.data:
            return False
        else:
            return match_tree(root1.left, root2.left) and match_tree(root1.right, root2.right)
    
    def sub_tree(root1, root2):
        if root1 is None:
            return False
        elif root1.data == root2.data and match_tree(root1, root2):
            return True
        else:
            return sub_tree(root1.left, root2) or sub_tree(root1.right, root2)
    
    if root1 is None or root2 is None:
        return False
    return sub_tree(root1, root2)


# Problem:4.11 Random Node: You are implementing a binary tree class from scratch which, in addition to
# insert, find, and delete, has a method getRandomNode() which returns a random node
# from the tree. All nodes should be equally likely to be chosen. Design and implement an algorithm
# for getRandomNode, and explain how you would implement the rest of the methods.
    '''
    Algorithm:
        Insert, Find and Delete methods are similar to standard BST methods.
        To do a random node search, we can traverse the whole tree and add all elements in a list and take a random 
        element from it but it will make the algorithm linear.
        Since we are implementing the class from the strach, we can maintain a size attribute in the node that 
        will have total number of nodes as size.
        Now, we will take a random number between 1 and root size.
        Say, root size is 8, root left size is 3 and root right size is 4.
        If random number is 8, return root.
        if random number is from 1 to 3 then we will choose a node from left subtree.
        Else, we will choose a node from right subtree. This way, every node will have the same probability for being 
        selected. 
        **** Refer avl_with_left_right_nodes_count.py Script in Repository "DATA_STRUCTURE_PY/TREES/BALANCED_BST/" ****
    '''


# Problem:4.12 Paths with Sum: You are given a binary tree in which each node contains an integer value (which
# might be positive or negative). Design an algorithm to count the number of paths that sum to a
# given value. The path does not need to start or end at the root or a leaf, but it must go downwards
# (traveling only from parent nodes to child nodes).
def paths_with_sum(root, sum, result=None):
    '''
        Algorithm:
            Brute Force Approach.
            We will start counting downward from each node and as soon as sum hits the given input, we will increase the 
            counter by 1.
            We will repeat this for each node and we will count till we hit the leaf.
    '''    
    def paths_with_sum_util(root, sum, result):
        if root is None:
            return
        if root.data == sum:
            result[0] += 1
        paths_with_sum_util(root.left, sum - root.data, result)
        paths_with_sum_util(root.right, sum - root.data, result)        

    if result is None:
        result = [0]
    paths_with_sum_util(root, sum, result)
    if root:    
        paths_with_sum(root.left, sum, result)  
        paths_with_sum(root.right, sum, result)  

    return result[0]
  

def paths_with_sum_optimized(root, sum):
    '''
    Algorithm:
        Time Complexity of Brute Force Approach is O(NlogN).
        In order to optimize it, we can use the approach of running sum and target sum.
        Say if we have an array and we need to find out number of ways where a sequence of elements can be summed to a target,
        then we can maintain a Hash table with running sum at each index. Then possible sequence on that index will be the 
        Hash Table value for difference of Running Sum and Target Sum.
        For ex: Running Sum at index 3 and 5 is 16. Running Sum at index 9 is 24. If we have a target sum of 8 then we will have 
        2 sequence of elements that will end at index 9 which will be summed to 8.
        Now, in order to apply similar logic in a Tree will be same except whenever we are done with a Node, we need to remove
        its Running Sum from the Hash Table.
        We will do Depth First Search on the Tree and implement this algorithm to get the total possible path counts.
    '''
    def paths_with_sum_optimized_util(root, target_sum, running_sum, path_count):
        if root is None:            
            return 0                
        running_sum += root.data
        difference = running_sum - target_sum
        try:
            total_paths = path_count[difference]
        except KeyError:
            total_paths = 0        
        # if Running Sum equals to Target Sum then 1 more path will be added in the count.
        if running_sum == target_sum:
            total_paths += 1
        
        try:
            path_count[running_sum] += 1
            if path_count[running_sum] == 0:
                del path_count[running_sum]
        except KeyError:
            path_count[running_sum] = 1
        
        total_paths += paths_with_sum_optimized_util(root.left, target_sum, running_sum, path_count)
        total_paths += paths_with_sum_optimized_util(root.right, target_sum, running_sum, path_count)
        
        path_count[running_sum] -= 1
        if path_count[running_sum] == 0:
            del path_count[running_sum]

        return total_paths

    path_count = {}
    return paths_with_sum_optimized_util(root, sum, 0, path_count)


if __name__ == "__main1__":
    print("Problem# 4.1")
    d = {
        'a': ['c'],
        'b': ['d'],
        'c': ['e', 'a'],
        'd': ['a', 'd'],
        'e': ['b', 'c'],
        'f': []
    }
    graph = Graph(d)
    print(route_between_nodes(graph, 'a', 'b'))

    print("\nProblem# 4.2")
    tree = create_min_height_bst([1,2,3,4,5,6])
    tree.print()

    print("\nProblem# 4.3")
    tree = create_min_height_bst([1,2,3,4,5,6,7])
    print_tree(tree.root)
    list_of_depths(tree.root)

    print("\nProblem# 4.4")
    tree = BTree()
    tree.root = Node(1)
    tree.root.right = Node(2)
    tree.root.left = Node(0)
    tree.root.right.right = Node(3)
    tree.root.right.left = Node(3)
    tree.root.right.left.left = Node(3)
    tree.root.right.left.left.left = Node(3)
    tree.print()
    print(is_balanced(tree.root))

    print("\nProblem# 4.5")
    tree = create_min_height_bst([0,1,2,3,1,6,7])
    tree.print()
    print(validate_bst(tree.root))
    tree.root.left.data = 10
    tree.print()
    print(validate_bst(tree.root))

    print("\nProblem# 4.6")
    get_successor()

    print("\nProblem# 4.7")
    d = {
        'a': ['d'],
        'b': ['d'],
        'c': [],
        'd': ['c'],
        'e': [],
        'f': ['b', 'a']
    }
    graph = Graph(d)
    print(graph)
    print()
    print(graph_topological_sort(graph))

    print("\nProblem# 4.8")
    tree = create_min_height_bst([1,2,3,4,5,6,7,8,9])
    tree.print()
    print(common_ancestor(tree.root, 4, 2))

    print("\nProblem# 4.9")
    tree = BTree()
    tree.root = Node(2.5)
    tree.root.left = Node(2)
    tree.root.right = Node(3)
    tree.root.left.left = Node(1)
    tree.root.right.right = Node(4)

    print_tree(tree.root, 2)
    print(all_bst_array_sequences(tree.root))

    print("\nProblem# 4.10")
    tree1 = create_min_height_bst([1,2,3,4,5])
    tree2 = BTree()
    tree2.root = Node(4)
    tree2.root.right = Node(5)
    # tree2.root.right.right = Node(6)
    tree1.print()
    tree2.print()
    print(check_subtree1(tree1.root, tree2.root))
    print(check_subtree2(tree1.root, tree2.root))

    print("\nProblem# 4.12")
    tree = create_min_height_bst([-1,-2,-3,0,1,1,3,4,1,2,-5])
    print_tree(tree.root, 2)
    print(paths_with_sum(tree.root, -1))
    print(paths_with_sum_optimized(tree.root, -1))

    tree = BTree()
    tree.root = Node(10)
    tree.root.left = Node(5)
    tree.root.right = Node(-3)
    tree.root.left.left = Node(3)
    tree.root.left.right = Node(1)
    tree.root.right.right = Node(11)
    tree.root.left.left.left = Node(3)
    tree.root.left.left.right = Node(-2)
    tree.root.left.right.right = Node(2)
    print_tree(tree.root, 2)
    print(paths_with_sum(tree.root, 8))
    print(paths_with_sum_optimized(tree.root, 8))