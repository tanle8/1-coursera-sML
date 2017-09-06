# Evaluating a Learning Algorithm
## Evaluating a Hypothesis
Once we have done some trouble shooting for errors in our predictions by:
-   Getting ***more*** `training examples`
-   Trying ***smaller*** `sets of features`
-   Trying ***additional*** `features`
-   Trying `polynomial features`
-   Increasing or decreasing $\lambda$

We can move on to evaluate our new hypothesis.

A `hypothesis` may have a *low* error for the `training examples` but still be *inaccurate* (because of overfitting). Thus, to evaluate a hypothesis, given a dataset of `training examples`, we can split up the data into two sets: a **training set** and a **test set**. Typical, the `training set` consists of **70%** of your data and the test set is the remaining **30%**.

The new procedure using these two sets is then:
1.  Learn $\Theta$ and minimize $J_{train}{\Theta}$ using the training set
2.  Compute the test set error $J_{test}(\Theta)$

### The test set error
1.  For linear regression: $J_{test}(\Theta) = \frac{1}{2m_{test}} \sum_{i = 1}^{m_{test}}(h_{\Theta} (x_{test}^{(i)}) - y_{test}^{(i)})^2$

2.  For classification ~ Misclassification error (a.k.a. 0/1 misclassification error):
$err(h_\Theta(x),y) = \begin{matrix} 1 & \text{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\\ 0 & \text otherwise \end{matrix}$

This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:
$\text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test})$

This gives us the proportion of the test data that was misclassified.

## Selecting Model and Train/ Validation/ Test Sets
Just because a learning algorithm fits a training set well, that does **not** mean it is a good hypothesis.
It could over fit and as a result your predictions on the test set would be poor. The error of your hypothesis as measured on the data set with which your trained the parameters will be lower than the error on any other data set.

Given many models with different polynomial degrees, we can use a **systematic** approach to identify the ***best*** function. In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

One way to break down our dataset into the three sets is:
-   Training set: 60%
-   Cross validation set: 20%
-   Test set: 20%

We can now calculate three separate `error values` for the three different sets using the following method:
1.  Optimize the parameters in $\Theta$ using the `training set` for each polynomial degree.
2.  Find the polynomial degree $d$ with the least error using the cross validation set.
3.  Estimate the generalization error using the `test set` with $J_test(\Theta^{(d)})$, ($d$ = theta from polynomial with lower error);

This way, the degree of the polynomial $d$ has not been trained using the test set.

## Bias vs. Variance
### Diagnosing Bias vs. Variance
In this section we examine the relationship between the degree of the polynomial $d$ and the underfitting or overfitting of our hypothesis.
-   We need to distinguish whether **bias** or **variance** is the problem contributing to bad predictions.
-   High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.

The training error will tend to **decrease** as we increase the degree $d$ of the polynomial.

At the same time, the cross validation error will tend to **decrease** as we increase $d$ up to a point, and then it will increase as $d$ is increased, forming a convex curve.

`High bias` **(underfitting)**: both $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ will be high. Also, $J_{CV}(\Theta) \approx J_{train}(\Theta)$.

`High variance` **(overfitting)**: $J_{train}(\Theta)$ will be low and $J_{CV}(\Theta)$ will be much greater than $J_{train}(\Theta)$

They are summarized in the figure below:
![Bias and Variance](/image/w6-1-1.png)

### Regularization and Bias/Variance
![Linear Regression with Regularization](/image/w6-1-2.png)

In the figure above, we see that as $\lambda$ increases, our fit becomes more rigid.
On the other hand, as $\lambda$ approaches $0$, we tend to **overfit** the data.
So how do we choose our parameter $\lambda$ to get it ***just right***?

In order to choose the model and the regularization term $lambda$, we need to:
1.  Create a list of lambdas (i.e. $\lambda \in \{ 0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24 \}$  )  

2.  Create a set of models with different degrees or any other variants.

3.  Iterate through the $\lambda$s and for each $\lambda$ go through all the models to learn some $\Theta$.

4.  Compute the **cross validation error** using the learned $\Theta$  (computed with $\lambda$) on the $J_{CV}(\Theta)$ **without** regularixation or $\lambda = 0$.

5.  Select the best combo that produces the **lowest error** on the cross validation set.

6.  Using the best combo $\Theta$ and $\lambda$, apply it on $J_{test}(\Theta)$ to see if it has a *good* generalization of the problem.

### Learning Curves
Training an algorithm on a very few

### Deciding what to do next
