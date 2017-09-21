# Dimensionality Reduction
## Terminology
-   `Dimensionality Reduction`
    Dimensionality reduction is the process of reducing the `number of features` in a `model`.
    This can be done by **removing** *redundant* or *irrelevant* `features` (feature selection) or by **finding** a *new*, *smaller*, `set of features` which are functions of one or more of your *original* `features` (feature extraction).
    A common dimensionality reduction (feature extraction) method is **Principal Component Analysis**.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Dimensionality_reduction)

-   `Feature Vector`
    A feature vector is a `vector` of numbers that represent ***properties*** of an object that we want to use in our prediction or learning task.
    For example, if we want to predict the price of a house using its area and the number of rooms it contains, we would use a *2-dimensional* `feature vector` to represent the house where one dimension would be the `area` of the house and the second dimension that `number of rooms`.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Feature_vector)

-   `Principal Component Analysis` (PCA)
    PCA is a dimensionality reduction method that (linearly) transforms a set of observations into a set of values of linearly uncorrelated variables called `principal components`.
    This transformation is defined such that the first principal component has the largest possible variance, it is the line such that the variance of the data points projected on this line is the largest of all such possible lines.
    Each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Principal_component_analysis)

-   `Covariance Matrix`
    A covariance matrix is a matrix where the element in the i, j position is the covariance between the $i^{th}$ and $j^{th}$ elements of a random vector.
    Intuitively, the element in the i, j, position tells us how much the $i^{th}$ element will change if we change the $j^{th}$ element slightly (and vice versa).
    A covariance matrix has several properties such as that it is always a positive-definite matrix.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Covariance_matrix)

-   `SVD`
    The singular value decomposition (SVD) is a factorization of a real or complex matrix.
    The singular value decomposition of an $m \times n$ matrix is a factorization of the form $U \Sigma V^{*}$ $U \Sigma V^{*}$ where $U$ and $V$ are othonormal (or unitary) matrices and $\Sigma$ is a diagonal matrix with non-negative real numbers on the diagonal.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Singular_value_decomposition)

-   `Normalization`
    Normalization refers to the process of shifting and rescaling features so that features take on approximately the same range of values.
    This is useful particularly in `gradient descent` where *badly* scaled `features` can cause gradient descent to converge ***slowly***.

## Motivation
### Motivation 1: Data Compression


### Motivation 2: Visualization

## Principal Component Analysis
### Principal Component Analysis Problem Formulation
### Principal Component Analysis Algorithm

## Applying PCA
### Reconstruction from Compressed Representation
### Choosing the Number of Principal Components
### Advice for Applying PCA
We can use PCA to speed up learning algorithms.
