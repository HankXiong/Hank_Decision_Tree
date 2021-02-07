# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:38:16 2021

@author: Hank Xiong
"""

import numpy as np
# import pandas as pd
from abc import ABC, abstractmethod
'''
iteams
1. allow select sample based row_idxs; DONE
2. allow randomly select feature set in each tree; DONE
3. allow for random split; DONE
4. my idea - proportion best split, maximum likelihood to find the best split - to make split less biaed. DONE
'''

class myBaseTreeClassifier(ABC):
    @abstractmethod
    def __init__(self, split,
                 metrics_type,
                 MAX_DEPTH,
                 MIN_SAMPLES_LEAF,
                 MAX_FEATURES = None,
                 random_state = None,
                 pn = np.exp(-1)):
        
        self.split = split
        self.metrics_type = metrics_type
        self.MAX_DEPTH = MAX_DEPTH
        self.MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF
        self.MAX_FEATURES = MAX_FEATURES
        self.random_state = np.random.RandomState(random_state)
        self.pn = pn
        #self.random_state2 = RandomState(random_state)
    
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def validate_input(self):
        pass


    def eval_metrics(self, y):
        '''
        evaluate the impurity score

        Parameters
        ----------
        y : array of class of shape (n_sample,)
            the array of classes used to evaluate impurity


        Returns
        -------
        float
            different metrics used to measure impurity

        '''
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
    '''A Decision Tree Classifier
        

        Parameters
        ----------
        split : {'best','random','proportion_best','max_like_best'}
            the split method for the tree
            'best': find the split point among available feature set that decreases the impurity the most
            'random': randomly choose 1 split point for each feature in the feature set, and use the one that decreases impurity most as the split point
            'proportion_best': choose a proportion of samples and use the best split point among them (local maximum)
            'max_like_best': The strategy used to maximize the probability to find the best split point (assume sequential coming and cannot go back),
                        more details can be seen in https://en.wikipedia.org/wiki/Secretary_problem
            The default is 'best'.
            
        metrics_type : {'Gini','Entropy','Misclassification'}
            the metrics used to measure impurity in the target
            The default is 'Gini'.
            
        MAX_DEPTH : Int
            The maximum depth of the tree
            The default is 1.
            
        MIN_SAMPLES_LEAF : Int or float
            if Int, it specifies the minimum required number of samples required to be a leaf node, a split point will 
            only be considered if it leaves at least MIN_SAMPLES_LEAF training samples in each of the left and right branches. 
            if float, then ceil(min_samples_leaf * n_samples) will be used
            The default is 1.
            
        MAX_FEATURES : {Int, float, "sqrt", "log2"}
            The number of features to consider when looking for the best split:

            If int, then consider max_features features at each split.
            If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
            If “sqrt”, then max_features=sqrt(n_features).
            If “log2”, then max_features=log2(n_features).
            If None, then max_features=n_features.
            
            The default is None.
            
        random_state :seed : {None, int, array_like, BitGenerator}, optional
            Random seed used to initialize the pseudo-random number generator or
            an instantized BitGenerator. 
            The default is None.
            
        pn : float, optional
            The proportion of samples in each tree used to find local maximum
            The default is np.exp(-1), which is the optimal proportion for "max_like_best".


        '''
    def __init__(self,split = 'best', metrics_type = 'Gini', 
                 MAX_DEPTH = 1, MIN_SAMPLES_LEAF = 1, 
                 MAX_FEATURES = None, random_state = None, pn = np.exp(-1)):
        super().__init__(split = split, metrics_type = metrics_type,
                         MAX_DEPTH = MAX_DEPTH, 
                         MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF,
                         MAX_FEATURES = MAX_FEATURES,
                         random_state = random_state,
                         pn = pn)

        
    def __str__(self):
        s = f'n: {self.n}, score: {self.cur_score}'
        if not self.is_leaf():
            s += f'\n\t score_if_split: {self.score_if_split}, split_col_idx: {self.col_idxs[self.split_col_idx]}, split_value:{self.split_value}'
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
        
        self.validate_input()
        
        if self.split == 'best':
            self.__best_split() 
        elif self.split == 'random':
            self.__random_split()
        elif self.split == 'proportion_best' or self.split == 'max_like_best':
            self.__semi_random_split()
        else:
            raise ValueError("no such split method "+ self.split)
        
        return self
    
    def __best_split(self):
        # split the tree into lhs and rhs 
        for c in range(self.c): 
            ## go through best split in each available column and update the tree property
            self.__find_column_best_split(self.tree_X[:,c],self.tree_y, c)
        
        print(self.__str__() + f', max_depth: {self.MAX_DEPTH}')
        
        self.__depth_first_tree_builder()
        

        return
    
    def __random_split(self):
        for c in range(self.c):
            x = self.tree_X[:,c]
            ## randomly select a point in each feature, choose the selected best to split
            xi = self.random_state.choice(x,size = 1)
            lhs_idxs, rhs_idxs = (x<=xi), (x > xi) 
            ln,rn = np.sum(lhs_idxs), np.sum(rhs_idxs)
            if ln < self.MIN_SAMPLES_LEAF or rn < self.MIN_SAMPLES_LEAF:
                continue
            lhs_y, rhs_y = self.tree_y[lhs_idxs], self.tree_y[rhs_idxs]
            split_score = ln / self.n * self.eval_metrics(lhs_y) + rn / self.n * self.eval_metrics(rhs_y)
            if split_score < self.score_if_split and split_score < self.cur_score:
                self.score_if_split,self.split_value, self.split_col_idx = split_score, xi, c
        print(self.__str__() + f', max_depth: {self.MAX_DEPTH}')
        
        self.__depth_first_tree_builder()
        
        return
    
    def __semi_random_split(self):
        for c in range(self.c):
            
            for_compare_n = round(self.pn * self.tree_X.shape[0]  )
            self.__find_column_proportion_best(self.tree_X[:,c], self.tree_y, c, for_compare_n )
        
        print(self.__str__() + f', max_depth: {self.MAX_DEPTH}')
        
        self.__depth_first_tree_builder()
        pass
    
    def __depth_first_tree_builder(self):
        
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
                                            random_state=state1,
                                            pn = self.pn).fit(self.X, self.y, lhs_idxs)
        self.rhs = myDecisionTreeClassifier(split = self.split,
                                            metrics_type =  self.metrics_type, 
                                            MAX_DEPTH = self.MAX_DEPTH - 1, 
                                            MIN_SAMPLES_LEAF = self.MIN_SAMPLES_LEAF,
                                            MAX_FEATURES=self.MAX_FEATURES,
                                            random_state=state2,
                                            pn = self.pn).fit(self.X, self.y, rhs_idxs)
        return 
    
    def  __find_column_best_split(self, x, y, c):
        # given a column of x, find the best split value and corresponding score
        # update score_if_split, split_col_idx, split_value, 
        x, y = self.order_xy(x,y)
        for i in range(0, self.n - self.MIN_SAMPLES_LEAF):
 
            ## no update score because cannot split
            if (i < self.MIN_SAMPLES_LEAF - 1) or (x[i] == x[i+1]):
                continue
            
            xi = x[i]
            lhs_y, rhs_y = y[x <= xi], y[x > xi]
            split_score = len(lhs_y) / self.n * self.eval_metrics(lhs_y) + len(rhs_y)/self.n * self.eval_metrics(rhs_y)
            if split_score < self.score_if_split and split_score < self.cur_score:
                self.score_if_split,self.split_value, self.split_col_idx = split_score, xi, c
        return
    
    def __find_column_proportion_best(self, x, y, c, for_compare_n ):
        # given a column x, randomly choose p proportion, find the best split point among them
        # if split = "porportion_best":
            # go with that
        # elif split = "max_like_best":
            # go through the rest sequentially
            # once find one better point, go with that
            # otherwise randomly pick one
        shuffled_idxs = self.random_state.permutation(len(x))
        shuffled_for_compare_x = x[shuffled_idxs[:for_compare_n]]
        
        for i in range(0, len(shuffled_for_compare_x) ):
            xi = shuffled_for_compare_x[i]
            lhs_y, rhs_y = y[x <= xi], y[x > xi]

            if (len(lhs_y) < self.MIN_SAMPLES_LEAF - 1) or (len(rhs_y) < self.MIN_SAMPLES_LEAF - 1):
                continue
            
            split_score = len(lhs_y)  / self.n * self.eval_metrics(lhs_y) + len(rhs_y)/self.n * self.eval_metrics(rhs_y)
            if split_score < self.score_if_split and split_score < self.cur_score:
                self.score_if_split,self.split_value, self.split_col_idx = split_score, xi, c
                
        if self.split == 'proportion_best':
            pass
        
        elif self.split == 'max_like_best':
            rest_x  = x[shuffled_idxs[for_compare_n:]]
            best_in_for_compare = True
            for i in range(0, len(rest_x)):
                xi = rest_x[i]
                lhs_y, rhs_y = y[x <= xi], y[x > xi]
                if (len(lhs_y) < self.MIN_SAMPLES_LEAF - 1) or (len(rhs_y) < self.MIN_SAMPLES_LEAF - 1):
                    continue
            
                split_score = len(lhs_y)  / self.n * self.eval_metrics(lhs_y) + len(rhs_y)/self.n * self.eval_metrics(rhs_y)
                if split_score < self.score_if_split and split_score < self.cur_score:
                    self.score_if_split,self.split_value, self.split_col_idx = split_score, xi, c
                    ## found a better one, go with this
                    best_in_for_compare = False
                    break
            
            if best_in_for_compare:
                xi = self.random_state.choice(x)
                lhs_y, rhs_y = y[x <= xi], y[x > xi]
                if (len(lhs_y) < self.MIN_SAMPLES_LEAF - 1) or (len(rhs_y) < self.MIN_SAMPLES_LEAF - 1):
                    self.score_if_split,self.split_value, self.split_col_idx = np.float('inf'), None, None
                else:
                    split_score = len(lhs_y)  / self.n * self.eval_metrics(lhs_y) + len(rhs_y)/self.n * self.eval_metrics(rhs_y)
                    self.score_if_split,self.split_value, self.split_col_idx = split_score, xi, c
        else:
            raise ValueError('no such split method '+ self.split)
        
        
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
           give the predicted probability of all classes for each observation in X
           
        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
           the input samples for predicting
        Returns
        -------
         array-like of shape (n_samples, n_unique_classes)

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
        '''
        predict the most like class for each observation of X

        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
           the input samples for predicting

        Returns
        -------
        array-like of shape (n_samples, )
            the predict class

        '''
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
    
    def validate_input(self):
        if self.pn > 0 and self.pn < 1:
            pass
        else:
            raise ValueError('pn should be within (0,1)')

        if self.MIN_SAMPLES_LEAF >= 1:
            self.MIN_SAMPLES_LEAF = int(self.MIN_SAMPLES_LEAF)
        elif self.MIN_SAMPLES_LEAF > 0 and self.MIN_SAMPLES_LEAF < 1:
            self.MIN_SAMPLES_LEAF = int(np.ceil(self.MIN_SAMPLES_LEAF * self.X.shape[0]))