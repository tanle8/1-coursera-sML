# Terminology
-    `Overfitting` describes a situation where the model learns to describe random error or noise instead of ***"real"*** meaningful relationships between features and the output.
        -   Overfitting often occurs when a model is too ***complex*** or when there is ***insufficient*** data.
        -   A consequence of overfitting is that the model will likely do poorly on examples it was not trained on and have high test error.
        -   One way to avoid overfitting is to use `regularization`.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Overfitting)

-   `Polynomial`
    A polynomial is a math expression that consist of variables that involve the addition, subtraction or product of such variables.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Polynomial)

-   `Regression`
    Regression (or regression analysis) is a statistical process for estimating the ***relationships*** among `variables`. Typically, there is an output or `dependent variable` (such as the price of a house) and one or more `predictors` or features (such as the area of a house or the number of bedrooms).
    Specifically, regression models estimate how the "typical" value of the output variable changes when one of the features is changed (but the others are fixed constant).
    A common example of regression analysis is linear regression analysis is linear regression where the dependent variable is modeled as a linear function of the features.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Regression_analysis)

-   `Regularization`
    In `regression` or `classification`, regularization refers to the process of ***adding*** *additional* terms to our `cost function`, often to introduce a preference for *simpler* models.
    Regularization is often used as a technique to ***prevent*** overfitting.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Regularization_(mathematics))

# Introduction
Machine learning models need to generalize well to ***new*** examples that the model has not seen in practice.

In this module, we introduce regularization, which helps prevent models from overfitting the training data.

## Solving the Problem of Overfitting

### The problem of Overfitting
Consider the problem of predicting $y$ from $x \in R$.
-   The leftmost figure below shows the result of fitting a $y = \theta_0 + \theta_1x$ to a dataset. We see that the data doesn't really lie on straight line, and so the fit is not very good.

![overfitting](/image/w3-2-1.png)
-   Instead, if we had added an extra feature $x^2$, and fit $y = \theta_0 + \theta_1x + \theta_2x^2$, then we obtain a *slightly* better fit to the data (See middle figure).
-   However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a $5^{th}$ order polynomial $y = \sum_{j = 0}^{5} \theta_jx^j$. We see that even though the fitted curve passes through the data perfectly, we would ***not*** expect this to be a very good predictor of, say, `housing prices` ($y$) for different `living areas` ($x$).

Without formally defining what these terms mean, we'll say the figure on the left shows an instance of `underfitting` - in which the data clearly shows structure not captured by the model - and the figure on the right is an example of `overfitting`.

`Underfitting`, or high bias, is when the form of our hypothesis function $h$ maps poorly to the trend of the data.

It is usually caused by a function that is too ***simple*** or uses too ***few*** features.
At the other extreme, `overfitting`, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a *complicated* function that creates a lot of *unnecessary* curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are ***2 main*** options to address the issue of overfitting:
1. **Reduce the number of features**:
    -   Manually select which features to keep.
    -   Use a model selection algorithm (studied later in the course).
2. **Regularization**
    -   Keep all the features, but reduce the magnitude of parameters $\theta_j$.
    -   *Regularization works well when we have a lot of slightly useful features*.

### Cost Function
If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by ***increasing*** their `cost`.

Say we wanted to make the following function more ***quadratic***:
$\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4$

We'll want to eliminate the influence of $\theta_3x^3$ and $\theta_4x^4$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our **cost function**:
$$\min_\theta \frac{1}{2m}\sum_{i = 1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\theta_3^2 + 1000\theta_4^2$$

We've added two *extra* terms at the end to **inflate** the cost of $\theta_3$ and $\theta_4$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_3$ and $\theta_4$ to near `zero`.
This will in turn greatly reduce the values of $\theta_3x^3$ and $\theta_4x^4$ in our hypothesis function.
As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but firs the data better due to the extra small terms $\theta_3x^3$ and $\theta_4x^4$.
![increase the cost function](/image/w3-2-2.png)

We should also regularize all of our `theta` parameters in a single summation as:
$$\min_\theta \frac{1}{2m} \sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j = 1}^n \theta_j^2$$

-   The $\lambda$ or `lambda`, is the **regularization parameter**.
    -   It determines how much the costs of our theta parameters are inflated.

Using the above **cost function** with the `extra summation`, we can smooth the output of our hypothesis function to reduce overfitting.
If lambda is chosen to be too ***large***, it may smooth out the function too much and cause `underfitting`. Hence, what would happen if $\lambda = 0$ or is too small ?

### Regularized `Linear Regression`
We can apply regularization to both `linear regression` and `logistic regression`.
We will approach linear regression first.

#### Gradient Descent
We will modify our gradient descent function to separate our $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.
$$\begin{aligned}
& \text{Repeat}\ \lbrace \\
& \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\
& \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\\ &
\rbrace
\end{aligned}$$
The term $\frac{\lambda}{m}\theta_j$ performs our regularization. With some manipulation our update rule can also be represented as:
$$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i = 1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

-   The first term in the above equation, $1 - \alpha\frac{\lambda}{m}$ will always be less than $1$.
Intuitively you can see it as ***reducing*** the value of $\theta_j$ by some amount on *every* update.
Notice that the second term is now exactly the same as it was before.

#### Normal Equation
Now let's approach regularization using the alternate method of the *non-iterative* normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:
$$
\begin{aligned}
& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \\
& \text{where}\ \ L = \begin{bmatrix}
    0 & & & & \\
    & 1 & & & \\
    & & 1 & & \\
    & & & \ddots & \\
    & & & & 1 \\
\end{bmatrix}
\end{aligned}$$

-   $L$ is a matrix with $0$ at the top left and 1's down the diagonal, with $0$'s everywhere else.
    -   It should have dimension $(n + 1) \times (n + 1)$
    -   Intuitively, this is the identity matrix (though we are not including $x_0$), multiplied with a single real number $\lambda$.

Recall that if $m < n$, then $X^TX$ is `non-invertible`. However, when we add the term $\lambda \cdot L$, then $X^TX + \lambda \cdot L$ becomes `invertible`.


### Regularized `Logistic Regression`
We can regularize logistic regression in a similar way that we regularize linear regression.
As a result, we can avoid overfitting.
The following image shows how the regularized function, displayed by the $\color{pink}{pink}$ line, is less likely to overfit than the non-regularized function represented by the $\color{blue}{blue}$ line:
![Regularized logistic regression](/image/w3-2-3.png)

#### Cost function
Recall that our cost function for logistic regression was:
$J(\theta) = - \frac{1}{m} \sum_{i = 1}^m [y^{(i)}log(h_{\theta}(x^{(i)}) + (1 - y^{(i)})log(1 - h_\theta(x^{(i)}))]$

We can regularize this equation by adding a term to the end:
$J(\theta) = - \frac{1}{m} \sum_{i = 1}{m} [y^{(i)}log(h_\theta(x^{(i)})) + (1 - y^{(i)})log(1 - h_\theta(x^{(i)}))]  + \frac{\lambda}{2m} \sum_{j = 1}^n \theta_j^2$
-   The second sum, $\sum_{j = 1}^m \theta_j^2$ **means to explicitly exclude** the bias term, $\theta_0$. i.e. the $\theta$ vector is indexed from `0` to `n`
(holding $n + 1$ values, $\theta_0$ through $\theta_n$), and this sum explicitly skips $\theta_0$, by running from `1` to `n`, skipping `0`. Thus, *when computing the equation, we should continuously update the two following equations*:
![Gradient Descent](/image/w3-2-4.png)
