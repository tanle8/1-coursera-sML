# Multivariate Linear Regression
## Terminology
-   `Feature Vector`
    A feature vector is a vector of numbers that represent *properties* of an `object` that we want to use in our prediction or learning task.
    For example, if we want to predict the price of a house using its area and the number of rooms it contains, we would use a 2-dimensional feature vector to represent the house where one dimension would be the area of the house and the second dimension that number of rooms.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Feature_vector)

-   `Inner Product`
    The inner product (or dot product) is an operation that takes two equal-length vectors and returns a single number.
    The inner product is usually defined as the sum of the products of the corresponding entries of the two vectors.
    [Link to wikipedia article](https://en.wikipedia.org/wiki/Dot_product)

-   Linear Regression
-   Regression
-   Training example
-   Training Set
-   `Polynomial`
    A polynomial is a math expression that consist of variables that involve the addition, subtraction of product of such variables.
-   `Polynomial Regression`
    Polynomial regression refers to a class of models where the output value is modeled to be a polynomial function of the input features.

## Multiple Features
Linear regression with multiple variables is also known as **multiple linear regression**.

We now introduce notation for equations where we can have any number of input variables.
    $$\begin{align}
    x_j^{(i)}
    &= \text{value of feature} j \text{in the }i^{th}\text{ training example} \\
    x^{(i)}& = \text{the input (features) of the }i^{th}\text{ training example} \\
    m &= \text{the number of training examples} \\
    n &= \text{the number of features}
    \end{align}$$

The multivariate form of the hypothesis function accommodating these multiple features is as follows:
    $$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$$

In order to develop intuition about this function, we can think about $\theta_0$ as the basic price of a house, $\theta_1$ as the price per square meter, $\theta_2$ as the price per floor, etc. $x_1$ will be the number of square meters in the house, $x_2$ the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:
    $$\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \\
    x_1 \\
    \vdots \\
    x_n
    \end{bmatrix} = \theta^T x
    \end{align*}$$
This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.


## Gradient Descent for Multiple variables
The gradient descent equation itself is generally the same form; we just have to repeat it  for our 'n' features:

$$\begin{align*} & \text{repeat until convergence:} \; \lbrace \\ \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\\ \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \\ \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \\ & \cdots \\ \rbrace \end{align*}$$

In other words:
$$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align*}$$

The following image compares gradient descent with one variable to gradient descent with multiple variables:
![Gradient Descent with multiple variables](/image/w2-1-1.png)


## Gradient Descent in Practice
### 1. Feature Scaling
We can speed up `gradient descent` by having each of  our input values in roughly the same range.
-   This is because $\theta$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate ***inefficiently*** down to the optimum when the variables are very **uneven**.

The way to prevent this is to modify the **ranges** of our `input variables` so that they are all roughly the same. Ideally:
    $$-1 \le x_{(i)} \le 1$$
or
    $$-0.5 \le x_{(i)} \le 0.5$$

There aren't  exact requirements; we are only trying to *speed things up*. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are **feature scaling** and **mean normalization**.
-   `Feature scaling` involves dividing the input values by the range (i.e. the `maximum value` minus the `minimum value`) of the input variables, resulting in a new range of just `1`.
-   `Mean normalization` involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.
To implement both of these techniques, adjust the `input values` as shown in this formula:
$$x_i := \frac{x_i - \mu_i}{s_i}$$
where:
    -   $\mu_i$ is the **average** of all the values for feature $(i)$
    -    $s_i$ is the standard deviation.

Note that dividing by the `range`, or dividing by the `standard deviation`, give ***different*** results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if $x_i$ represents housing prices with a range of (100 to 2000) and a mean value of 1000, then, $x_i := \frac{price - 1000}{1900}$.


### 2. Learning Rate
**`Debugging gradient descent`**
Make a plot with `number of iterations` on the x-axis. Now plot the `cost function`, $J(\theta)$ over the number of iterations of gradient descent. If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

**`Automatic convergence test`**
Declare convergence if $J(\theta)$ decreases by less than $E$ in one iteration, where $E$ is some small value such as $10^{-3}$. However in practice it's difficult to choose this threshold value.

![Making sure gradient descent is working correctly](/image/w2-1-2.png)

It has been proven that if learning rate $\alpha$ is sufficiently small, the $J(\theta)$ will decrease on every iteration.

![Making sure gradient descent is working correctly](/image/w2-1-3.png)
>To summarize:
>-   if $\alpha$ is too small: ***slow*** convergence.
>-   if $\alpha$ is too large: may ***no*** decrease on every iteration and this may not converge.


## Features and Polynomial Regression
We can improve our `features` and the form of our `hypothesis function` in a couple different ways.

We can **combine** multiple features into one. For example, we can combine $x_1$ and $x_2$ into a new feature $x_3$ by taking $x_1.x_2$.

### Polynomial Regression
Our `hypothesis function` need not be **linear** (a straight line) if that does not fit the data well.

We can **change the behavior** or **curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example,
-   if our hypothesis function is $h_{\theta} (x) = \theta_0 + \theta_1x_1$ then we can create additional features based on $x_1$, to get the ***quadratic*** function $h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_1^2 + \theta_3x_1^3$

-   In the ***cubic*** version, we have created new features $x_2$ and $x_3$ where $x_2 = x_1^2$ and $x_3 = x_1^3$.

-   To make it a ***square root*** function, we could do: $h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2\sqrt{x_1}$


One important thing to keep in mind is, if you choose your features this way then `feature scaling` becomes very important.
    e.g. if **$x_1$** has range `1 - 1000` then range of **$x_1^2$** becomes `1 - 1000000` and that of **$x_1^3$** becomes `1 - 1000000000`
