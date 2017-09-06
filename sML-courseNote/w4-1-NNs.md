### Terminology
-   `Neural Network`
    Neural networks are a type of machine learning model that build up *complex* **models** by connecting simple units (called `neurons`). Each unit is linked to other units in the network and influence the "activation" of these other units.

-   `Parameter Vector`
    The parameter vector of a **model** refers to a vector that capture some `relationship` between the ***input*** variables (features, representations) and the ***output*** of a model.
    The goal of machine learning is often to find values of the `parameter vector` of a model that describe the data set well.

-   `Training Set`
    Training set refers to the `set of examples` that a model is trained on.
    This is in contrast with the ***test set*** which is a set of examples *not* used to train a model but used to **evaluate** how well the model will perform on previously unseen examples.

-   `Sigmoid function`
    A sigmoid function is a function having an "S" shaped curve. Often, this term refers to the logistic function $f(x) = \frac{1}{1 + e^{-x}}$.

-   `Linear Function`
    A linear function is a function where the output is a polynomial of degree zero or one. For example, a linear function with 1 variable f(x) has the form $a * x + b$.

-   `Decision Boundary`
    A decision boundary is a `curve` or `surface` that divides the underlying space into *two* sets, one for each class. The classifier classifies all the points on one side of the decision boundary as belonging to one class and all those on the other side as belonging to the other class.

-   `Classification Problem`
    A classification problem is the problem of identifying which category (out of a set of categories) an example belongs to.
    Example of classification problems include classifying tumors as benign or malignant and classifying handwritten digits into 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.

-   `Multiclass Classification`
    In machine learning, multiclass or multinomial classification is the problem of classifying instances into one of the more than two classes ( classifying instances into one of the two classes is called binary classification).

-   `Regression`
    Regression (or regression analysis) is a `statistical process` for estimating the ***relationships*** among variables.
    Typically, there is an output or dependent variable (such as the price of a house) and one or more predictors or features

-   `Logistic Regression`
    Logistic regression is a binary classification model that predicts a binary outcome (such as whether a tumor is benign or malignant) based on some features (such as the size and density of cells etc).

-   `Feature Vector`
    A feature vector is a vector of numbers that represent properties of an object that we want to use in our prediction or learning task.
    For example, if we want to predict the price of a house using its area and the number of rooms it contains, we would use a 2-dimensional feature vector to represent the house where one dimension would be the area of the house and the second dimension that number of rooms.

## Neural networks
### Model Representations
Let's examine how we will represent a `hypothesis function` using neural networks.
At a very simple level, neurons are basically computational units that take `inputs` (**dendrites**) as *electrical* inputs (called "spikes") that are channeled to `outputs` (**axons**).
In our model, our dendrites are like the input features $x_1 ... x_n$, and the output is the result of our hypothesis function.
In this model, our $x_0$ input node is sometimes called the "bias unit". It is always equal to 1.
In neural network, we use the same logistic function as in classification, $\frac{1}{1 + e^{-\theta^Tx}}$, yet we sometimes call it a sigmoid (logistic) activation function. In this situation, our ***"theta"*** parameters are sometimes called `weights`

Visually, a simplistic representation looks like:

$$\begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ \end{bmatrix}\rightarrow\begin{bmatrix}\ \ \ \\ \end{bmatrix}\rightarrow h_\theta(x)$$

Our input note (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers".

In this example, we label these intermediate or "hidden" layer nodes $a_0^2 ... a_n^2$ and call them "activation units."

$a_i^{(j)} = "activation"$ of unit $i$ in layer $j$.
$\theta^{(j)} = $ matrix of weights controlling function mapping from layer $i$ to layer $j + 1$.

If we had one hidden layer, it would look like:
$$\begin{bmatrix}
    x_0 \\
    x_1 \\
    x_2 \\
    x_3
\end{bmatrix}
\rightarrow
\begin{bmatrix}
    a_1^{(2)} \\
    a_2^{(2)} \\
    a_3^{(2)}
\end{bmatrix}
\rightarrow
    h_\theta(x)$$

The value for each of the "activation" nodes is obtained as follows:

$\begin{aligned}
a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \\
a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \\
a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \\ h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \\
\end{aligned}$

This is saying that we compute our activation nodes by using a 3x4 matrix of parameters.
We apply each row of the parameters to our inputs to obtain the value for one activation node.
Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $\Theta^{(2)}$ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, $\Theta^{(j)}$.

The dimensions of these matrices of weights is determined as follows:

$\text{If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$.}$

The `+1` comes from the addition in $\Theta^{(j)}$ of the "bias nodes," $x_0$ and $\Theta_0^{(j)}$. In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:
![model representation](/image/w4-1-1.png)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension $\Theta^{(1)}$ is going to be 4x3 where $s_j = 2$ and $s_{j + 1} = 4$, so $s_{(j + 1)} \times (s_j + 1) = 4 \times 3$.

Note: if network has $s_j$ units in layer $j$, $s_{j + 1}$ units in layer $j + 1$, the $\theta^{(j)}$ will be of dimension $s_{j + 1} \times (s_j + 1)$.

To re-iterate, the following is an example of a neural network:
$\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \\ a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \\ a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \\ h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \\ \end{align*}$

In this section we'll do a vectorized implementation of the above function. We're going to define a new variable $z_k^{(j)}$ that encompasses the parameters inside our g function. In our previous example if we replaced by the variable $z$ for all the parameters we would get:
$\begin{align*}a_1^{(2)} = g(z_1^{(2)}) \\ a_2^{(2)} = g(z_2^{(2)}) \\ a_3^{(2)} = g(z_3^{(2)}) \\ \end{align*}$

In other words, for layer $j = 2$ and node $k$, the variable `z` will be:
$z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n$

The vector representation of $x$ and $z^j$ is:
$\begin{aligned}x = \begin{bmatrix}x_0 \\ x_1 \\\cdots \\ x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \\ z_2^{(j)} \\\cdots \\ z_n^{(j)}\end{bmatrix}\end{aligned}$

Setting $x = a^{(1)}$, we can rewrite the equation as:
$z^{(j)} = \Theta^{(j-1)}a^{(j-1)}$

We are multiplying our matrix $\Theta^{(j - 1)}$ with dimension $s_j \times (n + 1)$ (where $s_j$ is the number of our activation nodes) by our vector $a^{(j - 1)} with height $(n + 1)$. This gives us our vector $z^{(j)}$ with heights $s_j$. Now we can get a vector of our activation nodes for layer $j$ as follows:
$a^{(j)} = g(z^{(j)})$

Where our function $g$ can be applied `element-wise` to our vector $z^{(j)}$.

We can then add a `bias` unit (equal to 1) to layer $j$ after we have computed $a^{(j)}$. This will be element $a_0^{(j)}$ and will be equal to `1`. To compute our final hypothesis, let's first compute another $z$ vector:
$z^{(j + 1)} = \Theta^{(j)}a^{(j)}$
We get this final `z` vector by multiplying the next `theta matrix` after $\Theta^{(j - 1)}$ with the values of all the activation nodes we just got.
This last `theta matrix` $\Theta^{(j)}$ will have only **one row** which is multiplied by one column $a^{(j)}$ so that our result is a single number. We the get our final result with:
$h_\Theta (x) = a^{(j + 1)} = g(z^(j + 1))$

Notice that in this **last step**, between layer $j$ and layer $j + 1$, we are doing **exactly the same thing** as we did in `logistic regression`. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex *non-linear* hypotheses.
