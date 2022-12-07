# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:43:09 2022

@author: Daniel
"""

import networkx as nx
from typing import Type
import matplotlib.pyplot as plt

class union_find:
    
    def __init__(self):
        
        self.data = {}
        self.index = {}
        self.key_counter = 0
        
    def __str__(self):
        ret_str = "{}".format(self.data)
        return ret_str
        
    def getKey(self, item):
        return self.index[item]
    
    def getKeys(self):
        return self.data.keys()
    
    def get_largest_subtree(self):
        max_var = 0
        max_key = 0
        for subtree_key in self.data:
            cur = self.data[subtree_key][0].number_of_nodes()
            
            if (cur >= max_var):
                max_var = cur
                max_key = subtree_key
        
        return subtree_key
    
    def get_subtree_from_key(self, key):
        return self.data[key]
        
    def get_subtree(self, item):
        # print(self.index)
        # print(self.data)
        # print(item)
        curKey = self.getKey(item)
        return self.data[curKey][0]
    
    def get_root(self, item):
        return self.data[self.getKey(item)][1]
    
    def set_subtree(self, item, subtree: Type[nx.Graph], root):
        self.data[self.getKey(item)] = [subtree, root]
        
    def new_subtree(self, item, subtree: Type[nx.Graph], root):
        self.key_counter += 1
        self.index[item] = self.key_counter
        self.data[self.getKey(item)] = [subtree, root]
        
    # def insert_into(self, item, new_value):
    #     curKey = self.getKey(item)
    #     self.index[new_value] = curKey
    #     self.data[curKey] = self.data[curKey].union(new_value)
    
    
    def merge_by_key(self, key1, key2):
        
        try:
            if key1 == key2:
                raise ValueError('cannot merge with self')
        except (IndexError):
                exit('Could not merge.')
                
        self.data[key1][0] = nx.compose(self.data[key2][0], self.data[key1][0])
        self.data[key1][0].addEdge(self.data[key1][1], self.data[key2][1])
        
        print("Why are we here??")
        
        for item in self.data[key1][0].nodes:
            self.index[item] = key1
          
        del self.data[key2] 
    
    def merge(self, item1, item2, root1=None, root2=None):
        
        try:
            if item1 == item2:
                raise ValueError('cannot merge with self')
        except (IndexError):
                exit('Could not merge.')
        
        key1 = self.getKey(item1)
        key2 = self.getKey(item2)
        
        
        self.data[key1][0] = nx.compose(self.data[key2][0], self.data[key1][0])
        
        if not (root1 and root2):
            self.data[key1][0].add_edge(self.data[key1][1], self.data[key2][1])
        else:
            self.data[key1][0].add_edge(root1, root2)
        
        # print("Merging tree with {} and {}".format(item1, item2))
        # nx.draw_networkx(self.data[key1][0])
        # plt.show()
        # print("New Root: {}".format(self.data[key1][1]))
        
        for item in self.data[key1][0].nodes:
            self.index[item] = key1
        
        del self.data[key2]
        
    
        