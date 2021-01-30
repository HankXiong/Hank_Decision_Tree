# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:38:16 2021

@author: Hank Xiong
"""

import numpy as np
'''
TODO iteam
1 allow predict probability for each class, use idx to subsample for subtree, but not subsample directly
2 allow for random split
3. my idea
'''

class myDecisionTreeClassifier():
    def __init__(self, metrics_type = 'Gini', MAX_DEPTH = 1, MIN_SAMPLES_LEAF = 1):
        self.MAX_DEPTH = MAX_DEPTH
        self.MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF
        self.metrics_type = metrics_type
        
    def __str__(self):
        s = f'n: {self.n}, score: {self.cur_score}'
        if not self.is_leaf():
            s += f'\n\t score_if_split: {self.score_if_split}, split_col_idx: {self.split_col_idx}, split_value:{self.split_value}'
        return s
    
    def fit(self, X, y):
        ## consider all the features at each subtree
        self.X = X ## X should be n-c array
        self.y = y ## y should be n-1 array
        self.n = self.X.shape[0]
        self.c = self.X.shape[1]
        
        self.cur_score = self.eval_metrics(self.y) ## score of current tree given the sample y
        self.score_if_split = np.float64('inf') ## to be updated if it can be splitted

        # self.unique_classes = no.unique(self.y)
        # self.n_classes = len(self.unique_classes)
    
        self.split_col_idx = None
        self.split_value = None
        
        ## most shown class and its prob in current tree sample
        self.most_shown_class, self.most_class_prob = self.most_shown_class_with_prob(self.y)
        
        ## TODO
        self.best_split() 
        
        return self
    
    
    def best_split(self):
        # split the tree into lhs and rhs 
        for c in range(self.c): 
            ## go through best split in each column and update the tree property
            self.find_column_best_split(self.X[:,c],self.y, c)
        
        print(self.__str__() + f', max_depth: {self.MAX_DEPTH}')

        if self.is_leaf() : 
            return
        
        x = self.X[:,self.split_col_idx]
        lhs_tf , rhs_tf = (x <= self.split_value), (x > self.split_value)

        ## depth first build
        self.lhs = myDecisionTreeClassifier(self.metrics_type, self.MAX_DEPTH - 1, self.MIN_SAMPLES_LEAF).fit(self.X[lhs_tf,:], self.y[lhs_tf])
        self.rhs = myDecisionTreeClassifier(self.metrics_type, self.MAX_DEPTH - 1, self.MIN_SAMPLES_LEAF).fit(self.X[rhs_tf,:], self.y[rhs_tf])

        pass
    
    def find_column_best_split(self, x, y, c):
        # given a column of x, find the best split value and corresponding score
        # update score_if_split, split_col_idx, split_value, 
        x, y = self.__order_xy(x,y)
        for i in range(0, self.n - self.MIN_SAMPLES_LEAF):
 
            ## no update score because cannot split
            if i < self.MIN_SAMPLES_LEAF - 1 or x[i] == x[i+1]:
                continue
            
            xi = x[i]
            lhs_y, rhs_y = y[x <= xi], y[x > xi]
            split_score = ( i+1 ) / self.n * self.eval_metrics(lhs_y) + (self.n - i - 1)/self.n * self.eval_metrics(rhs_y)
            if split_score < self.score_if_split and split_score < self.cur_score:
                self.score_if_split,self.split_value, self.split_col_idx = split_score, xi, c
        return
    
    def __order_xy(self, x, y):
        ## x is a column of X
        idx_small_biger = np.argsort(x)
        return x[idx_small_biger], y[idx_small_biger]
    
    def is_leaf(self):
        is_leaf = self.score_if_split == np.float64('inf') or (self.cur_score == 0 )
        is_leaf = is_leaf or (self.MAX_DEPTH <= 0) or (self.n <= self.MIN_SAMPLES_LEAF)
        return is_leaf
    
    def eval_metrics(self, y):
        n = y.shape[0]
        unique_classes, counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        
        if n_classes == 1:
            return 0
        pi_s = counts / n
        
        if self.metrics_type == 'Gini':
            val = np.sum(pi_s * (1 - pi_s))
        elif self.metrics_type == 'Entropy':
            val = np.sum(-pi_s * np.log(pi_s))
        elif self.metrics_type == 'Misclassification':
            val = 1 - np.max(pi_s)
        else:
            raise ValueError('No metrics type ' + self.metrics_type)
            
        return val
    
    def most_shown_class_with_prob(self, y):
        
        n = y.shape[0]
        unique_classes, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        
        return unique_classes[idx], counts[idx] / n
        
    def predict(self, X):
        if len(X.shape) < 2:
            return np.array(self.__predict_row(X).most_shown_class )
            
        return np.array([self.__predict_row(xi).most_shown_class for xi in X])
    
    def __predict_row(self, xi):
        ## prediction after fitting the tree
        if self.is_leaf(): ## boundary condition
            return self
        
        to_tree = self.lhs if xi[self.split_col_idx] <= self.split_value else self.rhs
 
        return to_tree.__predict_row(xi)

    
    