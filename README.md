# Hank_Decision_Tree
Conventional decision tree algorithm with my ideas and customization added

In this project, I will rebuild traditional Decision Tree algorithm from scratch and add a new feature -- semi-randomly split node. It basically takes a proportion subsample 
find the split point decreasing the impurity the most in the subsample, then go through the rest samples, once it finds a new point that could decrease the impurity more then this 
will be the new split point, otherwise a random point from the rest sample will be used to split.

*This idea comes from the optimal strategy for finding you best partner given you will only meet n partner in total*
