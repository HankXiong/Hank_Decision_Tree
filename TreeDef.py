# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:38:16 2021

@author: Hank Xiong
"""

import numpy as np
# import pandas as pd
from abc import ABC, abstractmethod
from numpy.random import RandomState
'''
TODO iteam
2 allow for random split
3. my idea
'''

class myBaseTreeClassifier(ABC):
    @abstractmethod
    def __init__(self, split,
                 metrics_type,
                 MAX_DEPTH,
                 MIN_SAMPLES_LEAF,
                 MAX_FEATURES = None,
                 random_state = None):
        self.split = split
        self.metrics_type = metrics_type
        self.MAX_DEPTH = MAX_DEPTH
        self.MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF
        self.MAX_FEATURES = MAX_FEATURES
        self.random_state = RandomState(random_state)
        #self.random_state2 = RandomState(random_state)

    @abstractmethod
    def __str__(self):
        pass
    
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
    
    def order_xy(self, x, y):
        ## x is a column of X
        idx_small_biger = np.argsort(x)
        return x[idx_small_biger], y[idx_small_biger]
    
    @abstractmethod
    def fit(self,X,y, row_idx):
        pass
    @abstractmethod
    def predict(self, X, y):
        pass
    @abstractmethod
    def is_leaf(self):
        pass  
    
    
        
class myDecisionTreeClassifier(myBaseTreeClassifier):
    def __init__(self,split = 'best', metrics_type = 'Gini', 
                 MAX_DEPTH = 1, MIN_SAMPLES_LEAF = 1, 
                 MAX_FEATURES = None, random_state = None):
        
        super().__init__(split = split, metrics_type = metrics_type,
                         MAX_DEPTH = MAX_DEPTH, 
                         MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF,
                         MAX_FEATURES = MAX_FEATURES,
                         random_state = random_state)

        
    def __str__(self):
        s = f'n: {self.n}, score: {self.cur_score}'
        if not self.is_leaf():
            s += f'\n\t score_if_split: {self.score_if_split}, split_col_idx: {self.split_col_idx}, split_value:{self.split_value}'
        return s
    
    def is_leaf(self):
        is_leaf = self.score_if_split == np.float64('inf') or (self.cur_score == 0 )
        is_leaf = is_leaf or (self.MAX_DEPTH <= 0) or (self.n <= self.MIN_SAMPLES_LEAF)
        return is_leaf 
    
    def fit(self, X, y, row_idxs = None):
        ## consider all the features at each subtree
        
        self.X = X ## X should be n-c array
        self.y = y## y should be n-1 array
        self.unique_classes = np.unique(self.y)
        ## the row index of the subsample in current tree, should only be row number
        self.row_idxs = np.arange(self.X.shape[0]) if (row_idxs is None) else  np.array(row_idxs) 
        self.col_idxs = self.__generate_col_idxs()

        
        ## using X[row_idx,:] subsample to start the tree
        self.tree_X = self.X[self.row_idxs,:][:,self.col_idxs]
        self.tree_y = self.y[self.row_idxs]
        self.n = self.tree_X.shape[0]
        self.c = self.tree_X.shape[1]
                
        self.cur_score = self.eval_metrics(self.tree_y) ## score of current tree given the sample tree_y
        self.score_if_split = np.float64('inf') ## to be updated if it can be splitted


        self.probs = self.__class_with_prob(self.tree_y)
    
        self.split_col_idx = None
        self.split_value = None

        ## the full set class probabilitys of each observation in X
        self.classes, self.class_probs = None, None
        
        ## TODO
        self.best_split() 
        
        return self
    
    def best_split(self):
        # split the tree into lhs and rhs 
        for c in range(self.c): 
            ## go through best split in each available column and update the tree property
            self.find_column_best_split(self.tree_X[:,c],self.tree_y, c)
        
        print(self.__str__() + f', max_depth: {self.MAX_DEPTH}')

        if self.is_leaf() : 
            return
        
        x = self.tree_X[:,self.split_col_idx]
        lhs_idxs , rhs_idxs = self.row_idxs[x <= self.split_value] , self.row_idxs[x > self.split_value] 

        ## depth first build
        state1,state2 = self.random_state.randint(1e5,size = 2)
        self.lhs = myDecisionTreeClassifier(split = self.split,
                                            metrics_type =  self.metrics_type, 
                                            MAX_DEPTH = self.MAX_DEPTH - 1, 
                                            MIN_SAMPLES_LEAF = self.MIN_SAMPLES_LEAF,
                                            MAX_FEATURES=self.MAX_FEATURES,
                                            random_state=state1).fit(self.X, self.y, lhs_idxs)
        self.rhs = myDecisionTreeClassifier(split = self.split,
                                            metrics_type =  self.metrics_type, 
                                            MAX_DEPTH = self.MAX_DEPTH - 1, 
                                            MIN_SAMPLES_LEAF = self.MIN_SAMPLES_LEAF,
                                            MAX_FEATURES=self.MAX_FEATURES,
                                            random_state=state2).fit(self.X, self.y, rhs_idxs)

        return
    
    def  find_column_best_split(self, x, y, c):
        # given a column of x, find the best split value and corresponding score
        # update score_if_split, split_col_idx, split_value, 
        x, y = self.order_xy(x,y)
        for i in range(0, self.n - self.MIN_SAMPLES_LEAF):
 
            ## no update score because cannot split
            if (i < self.MIN_SAMPLES_LEAF - 1) or (x[i] == x[i+1]):
                continue
            
            xi = x[i]
            lhs_y, rhs_y = y[x <= xi], y[x > xi]
            split_score = ( i+1 ) / self.n * self.eval_metrics(lhs_y) + (self.n - i - 1)/self.n * self.eval_metrics(rhs_y)
            if split_score < self.score_if_split and split_score < self.cur_score:
                self.score_if_split,self.split_value, self.split_col_idx = split_score, xi, c
        return

    
    def __class_with_prob(self, y):
        n = y.shape[0]
        cur_unique_classes, cur_counts = np.unique(y, return_counts=True)
        if len(cur_unique_classes) <  len(self.unique_classes):
            prob_map = dict(zip(cur_unique_classes,cur_counts/n))
            probs = np.zeros( len(self.unique_classes) ) 
            for i,cl in enumerate(self.unique_classes):
                if cl in cur_unique_classes:
                    probs[i] = prob_map[cl]
        else:
            probs = cur_counts / n
        return  probs
    
    def predict_probability(self, X):
        '''
            n * number of unique class pandas dataframe
        Parameters
        ----------
        X : n * m numpy.array
            new feature matrix for prediction
        Returns
        -------
         pandas dataframe, n by number of classes

        '''
        if len(X.shape) < 2:
            probs = np.array([self.probs])
            #probs = probs.reshape( (X.shape[0],1) )
        else:
            probs = np.zeros( (X.shape[0], len(self.unique_classes)) )
            for i,xi in enumerate(X):
                probs[i,:] = self.__find_tree(xi).probs
        #probs = pd.DataFrame(probs, columns= self.unique_classes)
        return probs
        
    def predict(self, X):
        probs = self.predict_probability(X)
        if len(X.shape) < 2:
            return self.unique_classes[np.argmax(probs)]
        else:
            return self.unique_classes[np.argmax(probs,axis = 1) ]
    
    def __find_tree(self, xi):
        ## given one set of features, find the leaf it will be allocated to
        if self.is_leaf(): ## boundary condition
            return self
        
        to_tree = self.lhs if xi[self.col_idxs[self.split_col_idx]] <= self.split_value else self.rhs
 
        return to_tree.__find_tree(xi)
    
    def __generate_col_idxs(self):
        tot_col_idxs = np.arange(self.X.shape[1])
        if self.MAX_FEATURES is None:
            col_idxs = tot_col_idxs
        elif isinstance(self.MAX_FEATURES,int):
            col_idxs = self.random_state.choice(tot_col_idxs, size = max(1,min(self.MAX_FEATURES,self.X.shape[1])),replace = False )
        elif isinstance(self.MAX_FEATURES,float):
            col_idxs = self.random_state.choice(tot_col_idxs, size =  max(1,int(self.MAX_FEATURES * self.X.shape[1])), replace = False )
        elif isinstance(self.MAX_FEATURES, str):
            if self.MAX_FEATURES == 'log2':
                col_idxs = self.random_state.choice(tot_col_idxs, size = int(np.log2(self.X.shape[1])),replace=True  )
            elif self.MAX_FEATURES == 'sqrt':
                col_idxs = self.random_state.choice(tot_col_idxs, size = int(np.sqrt(self.X.shape[1])),replace=True  )
            else:
                raise ValueError('No such {self.MAX_FEATURES} choice')
        return col_idxs