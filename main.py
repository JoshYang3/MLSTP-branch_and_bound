# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:51:01 2022

@author: Daniel
"""
import csv
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import Type
from unionFind import union_find


def load_instances(file_path: str) -> list[list[list]]:
    """
    load_instances will load from a csv ith format specified by project assignment
    k
    edges, vertices
    u, v
    ....
    edges, vertices
    u, v
    ....

    Parameters
    ----------
    file_path : str
        csv path.

    Returns
    -------
    list[list[list]]
        list contains 
        number of instances
        number of edges
        edge [u, v].

    """
    instances = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        
        num_instances = next(reader)
        for instance in range(int(num_instances[0])):
            cur = next(reader)
            vertices, edges = cur[:2]
            current_instance = [[vertices, edges]]
            for i in range(int(edges)):
                current_instance.append(next(reader))
            instances.append(current_instance)
    
    return instances

def leaf_count(G: Type[nx.Graph]) -> int:
    """
    Counts number of leaves in a graph, vertices with degree 1

    Parameters
    ----------
    G : Type[nx.Graph]
        the graph, should be tree.

    Returns
    -------
    int
        number of leaves.

    """
    
    leaves = 0
    
    for vertex in list(G.nodes):
        if (G.degree[vertex] == 1):
            leaves += 1
            
    return leaves

def maximally_leafy_forest(G: Type[nx.Graph]) -> Type[union_find]:
    """
    maximally_leafy_forest generation for Lu-Ravi algorithm

    Parameters
    ----------
    G : Type[nx.Graph]
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    Subtrees = union_find()
    degrees = {}
    
    for vertex in list(G.nodes):
        T = nx.Graph()
        T.add_node(vertex)
        
        Subtrees.new_subtree(vertex, T, vertex)
        degrees[vertex] = 0
        
    for vertex in list(G.nodes):
        S_prime = []
        d_prime = 0
        
        for edge in G.edges(vertex):
            if((edge[1] not in Subtrees.get_subtree(vertex)) 
               and (Subtrees.getKey(edge[1]) not in S_prime)):
                d_prime = d_prime + 1
                
                #Insert subtrees[u] into S_prime
                S_prime.append(Subtrees.getKey(edge[1]))
                
        if (degrees[vertex] + d_prime >= 3):
            for subtree in S_prime:
                cur_subtree = Subtrees.get_subtree_from_key(subtree)
                
                Subtrees.merge(vertex, cur_subtree[1])
                degrees[vertex] = degrees[vertex] + 1
                degrees[cur_subtree[1]] = degrees[cur_subtree[1]] + 1
                
    
    # print(Subtrees)
    
    return Subtrees

def combine_forest(F: Type[union_find], G: Type[nx.Graph], debug=True) -> Type[nx.Graph]:
    #TODO: FIX
    
    root_key = F.get_largest_subtree()
    root_tree = F.get_subtree_from_key(root_key)[0]
     
    unmerged = True
    unmerged_key = None
    
    while unmerged:
                          
        for subtree in list(F.getKeys()):
            if (subtree == root_key):
                continue
            
            flag = False
            
            for node in F.get_subtree_from_key(subtree)[0].nodes:
                if (flag):
                    break
                for check_node in root_tree.nodes:
                    if G.has_edge(node, check_node) or G.has_edge(check_node, node):
                        F.merge(check_node, node, root1=check_node, root2=node)
                        flag = True
                        break
                    
            if (flag == False):
                unmerged = True
                unmerged_key = subtree
        
        if len(list(F.getKeys())) <= 1:
            unmerged = False
        
        if unmerged:
            root_key = unmerged_key
            root_tree = F.get_subtree_from_key(root_key)[0]
    
    return F.get_subtree_from_key(list(F.getKeys())[0])[0]
    

def solve_instance(instance, draw=False, debug=False):
    
    G = nx.Graph()
    
    G.add_edges_from(instance[1:])
    
    if draw:
        nx.draw_networkx(G)
        plt.show()
    
    
    BFS_Tree = nx.bfs_tree(G, '1')
    BFS_leaves = leaf_count(BFS_Tree)
    
    F = maximally_leafy_forest(G)
    
    if debug:
        for tree_key in F.getKeys():
            nx.draw_networkx(F.get_subtree_from_key(tree_key)[0])
            plt.show()
    
    F_tree = combine_forest(F, G)
    F_tree_leaves = leaf_count(F_tree)
    
    if draw:
        nx.draw_networkx(F_tree)
        plt.show()
    
    print("BFS leaves: {}, Lu-Parv leaves: {}".format(BFS_leaves, F_tree_leaves))
    
    if(F_tree_leaves > BFS_leaves):
        return (F_tree, F_tree_leaves)
    
    else:
        return (BFS_Tree, BFS_leaves)
    

def run_instances(instances, file_name="all-solved.out"):
    for i, instance in enumerate(instances):
        
        print("Start instance: {}".format(i))
        try:
            T, leaves = solve_instance(instance)
            
            outlist = []
            
            head = [leaves, 0]
            
            for edge in T.edges:
                outlist.append(list(map(int, edge)))
                head[1] += 1
                
            outlist = sorted(outlist, key=lambda x: x[0])
            
            outlist = [head] + outlist
            
            with open(file_name, "a", newline='') as f:
               writer = csv.writer(f)
               
               writer.writerows(outlist)

        except (KeyError):
            print("Error with instance: {}".format(i))
            exit(1)
        
        print("Finished instance: {}".format(i))
    
    
def check_instances(instances) -> list:
    
    instance_out = []
    
    for i, instance in enumerate(instances):
        G = nx.Graph()
        
        G.add_edges_from(instance[1:])
        
        if (nx.is_connected(G)):
            instance_out.append(instance)
        else:
            print("Failure to load instance #{}".format(i))
    
    return instance_out

if __name__=="__main__":
    
    instances = load_instances(os.path.join(os.getcwd(), "all-hard.txt"))
    
    instances = check_instances(instances)
    
    run_instances(instances)
    
    
    
    
    
    
    
    
    