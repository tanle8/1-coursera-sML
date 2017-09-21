# Clustering
## Terminology
-   `Clustering Algorithm`
    **Clustering analysis** or **clustering** is the problem of ***grouping*** a `set of objects` so that objects in the *same* group (cluster) are more *similar* (using some similarity measure) to each other than to those in other groups.
    A common form of clustering is `K-means clustering` where the objects that are clustered are ***n-dimensional*** **vectors** and the "similarity" measure used is the `Euclidean distance` between **points** (the *closer* two points are, the more *similar* they are).
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Cluster_analysis)

-   `Decision Boundary`
    A decision boundary is a **curve** or **surface** that divides the underlying space into **two sets**, one for each class.
    The classifier classifies all the points on one side of the decision boundary as belonging to one class and all those on the other side as belonging to the other class.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Decision_boundary)

-   `Supervised Learning`
    Supervised learning is the machine learning task of learning a **way** to `map` `input signals` to `output values` by **training a model** on a `set of training examples` where each example is a pair consisting of an input and a desired output value (the label)
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Supervised_learning)

-   `Training Set`
    Training set refers to the `set of examples` that a model is trained on,
    This is in contrast with the **test set** which is a set of examples not used to train a model but used to `evaluate` how well the model will perform on previously unseen examples.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Test_set)

-   `Unsupervised Learning`
    Unsupervised machine learning is the machine learning task of inferring a function to describe hidden structure from "unlabeled" data.
    Examples of unsupervised learning tasks include clustering (where we try to discover underlying groupings of examples) and anomaly detection (where we try to infer if some examples in a dataset do not conform to some expected pattern).
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Unsupervised_learning)

-   `Cluster Assignment`
    Cluster assignment refers to the **process** (or **result**) of **grouping examples** in a `data set` into *disjoint* **groups** (called **clusters**).
    [Link to wikipedia article](https://en.wikipedia.org/wiki/K-means_clustering)

-   `Cluster Centroid`
    **Cluster centroid** refers to the *average* position of points belonging to a cluster.
    Specifically, in `K-means clustering`, the centroid of cluster $X$ refers to the **position** that ***minimizes*** the `sum of squared distances` to **all points** in cluster $X$.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/K-means_clustering)

-   `Clustering Algorithm`
    **Cluster analysis** or **clustering** is the problem of grouping a `set of objects` so that objects in the same group (cluster) are more similar (using some similarity measure) to each other than to those in other groups.
    A common form of clustering is `K-means` clustering where the objects that are clustered are **n-dimensional vectors** and the "similarity" measure used is the `Euclidean distance` between points (the closer two points are, the more similar they are).
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Cluster_analysis)

## Unsupervised Learning: Introduction
The given data is the set of examples with no labels, so our training set is just: $\{x^{(1)}, x^{(2)}, x^{(3)}, ... , x^{(m)} \}$
![Unsupervised Learning](/image/w8-1-1.png)

So, what is clustering good for?
One application is market segmentation where you may have a database of customers and want to group them into different market segments so you can sell to them separately or serve your different market better. Social network analysis, there are actually groups have done this things like looking at a group of people's social networks information about who other people that you email the most frequently and who are the people that they email most frequently and to find the coherence in groups of people. Or use clustering to organize computer cluster or to organize data centers better because if you know which computers in the data center in the cluster tend to work together, you can use that to reorganize your resources and how you layout the network and how you design your data center communications. Other application like using clustering algorithms to understand galaxy formation and using that to understand astronomical data.

## K-Means Algorithm
In the clustering problem, we are given an unlabeled dataset and we would like to have an algorithm automatically group the data into coherent subsets/ clusters for us.

> 1.  Place $K$ points into the space represented by the objects that are being clustered. These points represent *initial* group `centroids`.
> 2.  ***Assign*** *each* `object` to the group that has the ***closest*** `centroid`.
> 3.  When all objects have been assigned, **recalculate** the positions of the $K$ `centroids`.
> 4.  Repeat `Steps 2` and `3` until the `centroids` no longer move. This produces a separation of the objects into groups from which the metric to be minimized can be calculated.



## Optimization Objective
## Random Initialization
## Choosing the Number of Clusters
