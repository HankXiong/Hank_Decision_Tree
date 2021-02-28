# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:38:45 2021

@author: Hank Xiong
"""

from SemiRandomClassifier import SemiRandomDecisionTreeClassifier
import numpy as np

class SemiRandomForestClassifier():
    '''A Random Forest Classifier based on SemiRandom Decision Tree
    

    Parameters
    ----------
    n_trees : Int
        The number of trees to be built
        Default is 50.
        
    split : {'best','random','proportion_best','max_like_best'}
        the split method for the tree
        'best': find the split point among available feature set that decreases the impurity the most
        'random': randomly choose 1 split point for each feature in the feature set, and use the one that decreases impurity most as the split point
        'proportion_best': choose a proportion of samples and use the best split point among them (subsample maximum)
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
        for split method 'proportion_best' and 'max_like_best' only
        The proportion of samples in each tree used to find local maximum
        The default is np.exp(-1), which is the optimal proportion for "max_like_best".
        
    bootstrap: Bool
        If the sample fed into each tree should be bootstrap or not
        Default is True.
        
    bootstrap_max_samples: Int, float or None
        If bootstrap is true, this parameter determines the bootstrapped sample size
        if int, then each sample size would be bootstrap_max_samples.
        if float, then each sample sould be a proportion of the input sample.
        if None, then will use sample size equal to the input sample
        

    '''
        
    def __init__(self, n_trees = 50,
                  split = 'best',
                 metrics_type = 'Gini',
                 MAX_DEPTH = 3,
                 MIN_SAMPLES_LEAF = 1,
                 MAX_FEATURES = None,
                 random_state = None,
                 pn = np.exp(-1), 
                 bootstrap = True,
                 bootstrap_max_samples = None):
        
        self.split = split
        self.metrics_type = metrics_type
        self.MAX_DEPTH = MAX_DEPTH
        self.MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF
        self.MAX_FEATURES = MAX_FEATURES
        self.random_state = np.random.RandomState(random_state)
        self.pn = pn
        
        self.n_trees = n_trees
        self.bootstrap = bootstrap
        self.bootstrap_max_samples = bootstrap_max_samples
        
        self.trees = None
    def fit(self, X, y):
        '''
        fit a bunch of decision tree classifiers from the training set (X, y)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples
        y : array-like of shape (n_samples,)
            The target values as itegers or strings

        Returns
        -------
        self
            fitted random forest classifier with attribute trees containing all fitted decision tree
           

        '''
        self.X = X
        self.y = y
        self.unique_classes = np.unique(self.y)
        self.__validate_input()
        sub_random_states = self.random_state.choice( int(1e6), size = self.n_trees, replace = True)
        
        trees = []
        for i in range(self.n_trees):
            print("------ This is the %dth tree-----" % (i+1)  )
            if self.bootstrap:
                boot_row_idxs = np.random.RandomState(sub_random_states[i]).choice(X.shape[0],size = self.bootstrap_max_samples,
                                                                                  replace=True)
                boot_X,boot_y = X[boot_row_idxs,:], y[boot_row_idxs]
                trees.append( self.__fit_single_tree(boot_X, boot_y, sub_random_states[i]) )
            else:
                trees.append(self.__fit_single_tree(X, y, sub_random_states[i]))
        
        self.trees = trees
        return self
         
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
        probs = np.zeros( (X.shape[0], len(self.unique_classes) ) )
        for k,tree in enumerate(self.trees):
            probs = ( probs * k + tree.predict_probability(X) ) / (k+1) ## interation to have the average
            
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

    def __fit_single_tree(self, boot_X, boot_y, sub_state):
        
        fitted_tree = SemiRandomDecisionTreeClassifier(split = self.split,
                                                       metrics_type=self.metrics_type,
                                                       MAX_DEPTH= self.MAX_DEPTH,
                                                       MIN_SAMPLES_LEAF=self.MIN_SAMPLES_LEAF,
                                                       MAX_FEATURES=self.MAX_FEATURES,
                                                       random_state=sub_state,
                                                       pn = self.pn).fit(boot_X,boot_y)
        return fitted_tree
                                                       
                                                       
    def __validate_input(self):
        if self.n_trees < 1:
            raise  ValueError("number of trees should be larger than or equal to 1")
        else:
            self.n_trees = int(self.n_trees)
            
        if self.bootstrap_max_samples is None:
            self.bootstrap_max_samples = self.X.shape[0]
        elif isinstance(self.bootstrap_max_samples, int):
            self.bootstrap_max_samples = int(self.bootstrap_max_samples)
        elif isinstance(self.bootstrap_max_samples,float) and self.bootstrap_max_samples > 0:
            self.bootstrap_max_samples = int(np.round(self.bootstrap_max_samples * self.X.shape[0]))
        else:
            raise ValueError("No such bootstrap_max_samples choice")
            

        pass