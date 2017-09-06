# Terminology
-   `Liner Regression`
    Linear Regression is a model for modeling the ***relationship*** between a `numerical variable` y and one or more `explanatory variables` (or features) as a linear function.

-   `Logistic Regression`
    Logistic Regression is a *binary* **classification model** that predicts a `binary outcome` (such as whether a tumor is benign or malignant) based on some `features` (such as the size and density of cells, etc)

-   `Classification Problem`
    A classification problem is the problem of identifying which **category** (out of a **set of categories**) an example belong to.
    Example of classification problems include classifying tumors as benign or malignant and classifying handwritten digits into 0 to 9.

-   `Multi-class Classification`
    In machine learning, ***multi-class*** or ***multinomial*** classification is the problem of classifying `instances` into one of the more than two `classes` (classifying instances into one of the two classes is called binary classification)

-   `Norm`
    A norm is a ***function*** that defines some `notion of length`.
    Specially, a norm is a function that ***maps*** an `input` to a **strictly** positive number, other than a special input called the zero input.
    Norms must also satisfy linearity properties.
    The most common example of a norm is the `Euclidean norm` which is our typical notion of the length of a vector.


-   Regression

-   `Cost function`
    In machine learning, the cost function is a function that maps parameters of a model to a number (the cost).
    The cost function measures how well or how badly a model fits the dataset and the objective is often to find a model that describes the data well by picking a model that minimizes some cost function.
    Examples of common cost functions include the **squared error loss** (used in linear regression) or the **logistic loss** (used in logistic regression).

-   `Parameter vector`
    The parameter vector of a model refers to a vector that capture some relationship between the input variables (features) and the output of a model.
    The goal of machine learning is often to find values of the parameter vector of a model that describe the data set well.

-   `Derivative`
    The derivative of a function measures the sensitivity of the function value with respect to a change in its argument (input value).
    Intuitively, the derivative tells us how much the function value will change if we change the input values slightly.

-   `Optimization algorithm`
    An Optimization algorithm is a `process` for *maximizing* or *minimizing* a real function by systematically ***choosing*** `input values` and ***evaluating*** the value of the function at that `point`.
    Example of optimization algorithms include gradient descent or Newton's method.

-   `Parameter Vector`
    The parameter vector of a model refers to a `vector` that ***capture*** some `relationship` between the `input variables` (features) and the `output` of a model.
    The goal of machine learning if often to **find** values of the parameter vector of a model that describe the data set well.

-   Partial Derivative
    A partial derivative of a function is its `derivative` (gradient) with respect to one of its argument/variables with all other variables kept constant.
    We can think of the partial derivative as how much we expect the value of a function to change if we make a small change to  one of its arguments (leaving the other arguments unchanged)

# Logistic Regression
## Classification and Representation
### Classification
To attempt classification, one method is to use `linear regression` and map all predictions greater than 0.5 as a `1` and all less than 0.5 as `0`. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values `y` we now want to predict take on only a small number of discrete values.
For now, we will focus on the **binary** classification problem in which `y` can take on only two values, **0** and **1** (Most of what we say here will also generalize to the multiple-class case).
-   For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some features of a piece of email, and `y` may be *1* if it is a piece of spam mail, and *0* otherwise. Hence, $y \in  {0,1}$. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols `-` and `+`. Given $x^{(i)}$, the corresponding $y^{(i)}$ is also called the ***label*** for the training example.


### Hypothesis Representation
[Video](https://www.coursera.org/learn/machine-learning/lecture/RJXfB/hypothesis-representation)
We could approach the classification problem ignoring the fact that $y$ is ***discrete-valued***, and use our old `linear regression` algorithm to try to predict y given x.
However, it is easy to construct examples where this method performs very poorly.
Intuitively, it also doesn't make sense for $h_\theta (x)$ to take values larger than 1 or smaller than 0 when we know that $y \in {0,1}$.
To fix this, let's change the form for our hypotheses $h_\theta (x)$ to satisfy $0 \le h_\theta (x) \le 1$. This is accomplished by plugging $\theta^Tx$ into the Logistic Function.

Our new form uses the `Sigmoid Function`, also called the **Logistic Function**:
> $h_\theta (x) = g(\theta^Tx)$
> $z = \theta^Tx$
> $g(z) = \frac{1}{1 + e^{-z}}$

The following image shows us what the sigmoid function looks like:
![sigmoid function](/image/w3-1-1.png)

The function $g(z)$, shown here, maps any real number to the (0, 1) interval, making it useful for transforming an ***arbitrary-valued*** function into a function better suited for classification.

$h_\theta (x)$ will give us the probability that our output is 1.
-   For example, $h_\theta (x)$ = 0.7 gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).
> $h_\theta (x) = P(y = 1 | x; \theta) = 1 - P(y = 0 | x; 0)$
> $P(y = 0 | x; \theta) + P(y = 1 | x; \theta) = 1$

### Decision Boundary

## Logistic Regression Model
### Cost Function
We cannot use the same cost function that we use for linear regression that we use for linear regression because the Logistic Function will cause the output to be ***wavy***, causing many *local* optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:
> $$J(\theta) = \frac{1}{m}\sum_{i = 1}^m Cost(h_\theta(x^{(i)}), y^{(i)})$$
> $Cost(h_\theta(x), y) = - log(h_\theta(x)) \qquad \qquad if \ y = 1$
> $Cost(h_\theta(x), y) = -log(1 - h_\theta(x)) \qquad if \ y = 0$

When $y = -1$, we get the following plot for $J(\theta)$ vs $h_\theta(x)$:
![](/image/w3-1-2.png)

Similarly, when $y = 0$, we get the following for $J(\theta)$ vs $h_\theta(x)$:
![](/image/w3-1-3.png)

>$Cost(h_\theta(x), y) = 0 \qquad if \ h_\theta(x) = y$
$Cost(h_\theta(x), y) \to \infty \qquad if \ y = 0 \ and \ h_\theta(x) \to 1$
$Cost(h_\theta(x), y) \to \infty \qquad if \ y = 1 \ and \ h_\theta(x) \to 0$

if our *correct* answer $y$ is 0, then the cost function will be 0 if our hypothesis function also outputs 0.
-   If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1.
-   If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that $J(\theta)$ is ***convex*** for logistic regression.

### Simplified Cost Function and Gradient Descent

### Advanced Optimization
Optimization algorithms:
-   Gradient Descent
-   Conjugate Gradient
-   BFGS
-   L-BFGS

Advantages:
-   No need to manually pick $\alpha$
-   Often faster than gradient descent.

Disadvantages:
-   More complex

"Conjugate gradient", "BFGS", and "L-BFGS" are more *sophisticated*, *faster* ways to optimize $\theta$ that can used instead of gradient descent.
We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized.

We first need to provide a function that evaluates the following two functions for a given input value $\theta$:
$J(\theta)$
$\dfrac{\partial}{\partial \theta_j}J(\theta)$

We can write a single function that returns both of these:

```Matlab
function [jVal, gradient] = costFunction(theta)
    jVal = % code to compute J(theta)
    gradient(1) = CODE#1 % derivative for theta_0
    gradient(2) = CODE#2 % derivative for theta_1
```
```Matlab
function [jVal, gradient] = costFunction(theta)

jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2;

gradient = zeros(2,1);
gradient(1) = 2*(theta(1) - 5);
gradient(2) = 2*(theta(2) - 5);
```
Then we can use octave's `fminunc()` optimization algorithm along with the `optimset()` function that creates an object containing the options we want to send to `fminunc()`.

```Matlab
options = optimset('GradObj', 'on', 'MaxIter', '100');
initialTheta = zeros(2,1);
    [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

-   `'GradObj', 'on'`: set Gradient objective parameter to *on*. This means you are indeed going to provide a gradient to this algorithm.
-   `'MaxIter', '100'`: Set maximum number of iteration to *100*.
-   `initialTheta = zeros(2,1)`: give the function an initial guess for theta (a 2-by-1 vector).
-   `optimset('GradObj', 'on', 'MaxIter', '100');
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] ...
    = fminunc(@costFunction, initialTheta, options);` - call the `fminunc` function.
    -   `@` represent a pointer to the cost function.

We give the function `fminunc()` our cost function, our initial vector of theta values, and the `option` object the we created beforehand.


## Multi-class Classification
In machine learning. multi-class or multinomial classification is the problem of classifying instances into one of the more than two classes (classifying instances into one of the two classes is called binary classification).
[Link to wikipedia article](https://en.wikipedia.org/wiki/Multiclass_classification)

### Multi-class Classification: One-vs-All
Now we will approach the classification of data when we have more than two categories. Instead of $y = {0,1}$ we will expand our definition so that $y = \{0, 1, ... n\}$.

Since $y = \{0, 1, ... n\}$, we divide our problem into $n + 1$ ($+1$ because the index starts at 0) binary classification problems; in each one, we predict the probability that $y$ is a member of one of our classes.

$\begin{aligned}
y \in \{0, 1, ... n\}\\
h_\theta^{(0)} (x) = P(y = 0 | x; \theta)\\
h_\theta^{(1)} (x) = P(y = 1 | x; \theta)\\
...\\
h_\theta^{(n)} = P(y = n | x; \theta)\\
prediction = \max_i(h_\theta^{(i)}(x))
\end{aligned}$

We are basically choosing one class and then lumping all the others into a single second class.
We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

The following image shows how one could classify 3 classes:
![example of classification](/image/w3-1-4.png)

**To Summarize**:
-   Train a logistic regression classifier $h_\theta(x)$ for each class to predict the probability that $y = i$.
-   To make a prediction on a new $x$, pick the class that maximizes $h_\theta(x)$


## Review
[Lecture Slides](https://www.coursera.org/learn/machine-learning/supplement/QEYX8/lecture-slides)
