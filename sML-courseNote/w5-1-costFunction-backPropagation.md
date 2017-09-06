# Cost Function and Backpropagation
## Terminology
`Neural Network`
Neural Networks are a type of machine learning model that build up complex models by connecting simple units (called neurons).
Each unit is linked to other units in the network and influence the "activation" of these other units.
[Link to wikipedia article](https://en.wikipedia.org/wiki/Artificial_neural_network)
`Backpropagation`
Backpropagation is a method to calculate the **gradient** of the loss function with respect to the weights of an artificial neural network.
It is used in conjunction with optimization algorithms such as gradient descent to train neural networks.
When an input vector is presented to the network, it is propagated forward through the network, layer by layer, until it reaches the output layer.
Backpropagation then takes the gradient of the loss function with respect to the output of the final layer and "propagates" *backwards* from the output layer using the `chain rule`.
These gradients are the used to calculate the gradient of the loss function with respect to the weights of the network.
[Link to wikipedia article](https://en.wikipedia.org/wiki/Backpropagation)


## Cost Function
Let's first define a few variables that we will need to use:
-   $L$ = total number of layers in the network
-   $s_l$ = number of units (not counting bias unit) in layer $l$.
-   $K$ = number of output units/classes

Recall that in neural networks, we may have *many* output nodes.
We denote $h_\Theta(x)_k$ as being a `hypothesis` that results in the $k^{th}$ `output`.
Our `cost function` for **neural networks** is going to be a ***generalization*** of the one we used for logistic regression. Recall that the `cost function` for *regularized* **logistic regression** was:
$$J(\theta) = -\frac{1}{m}\sum_{i = 1}^m[y^{(i)}log(h_\theta(x^{(i)})) + (1 - y^{(i)})log(1 - h_\theta({x^{(i)}}))] + \frac{\lambda}{2m}\sum_{j = 1}^{n}\theta_j^2$$

For `neural networks`, it is going to be slightly more complicated:

$$J(\Theta) = -\frac{1}{m}\sum_{i = 1}^{m}\sum_{k = 1}^{K}[y_k^{(i)}log((h_\Theta(x^{(i)}))_k) + (1 - y_k^{(i)})log(1 - (h_\Theta(x^{(i)}))_k)] + \frac{\lambda}{2m}\sum_{l = 1}^{L -1}\sum_{i = 1}^{s_l}\sum_{j = 1}^{s_l + 1}(\Theta_{j,i}^{(l)})^2$$

We have added a few *nested* summations to account for our ***multiple*** output nodes. In the first part of the equation, before the square brackets, we have an *additional* nested summation that ***loops*** through the `number of output nodes` (K units/classes).

In the regularization part, after the square brackets, we must account for *multiple* `theta matrices`.
-   The `number of columns` in our ***current*** `theta matrix` is equal to the `number of nodes` in our ***current*** layer (including the bias unit).
-   The `number of rows` in our ***current*** `theta matrix` is equal to the `number of nodes` in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

Note:
-   The ***double sum*** simply adds up the logistic regression costs calculated for each cell in the output layer.

-   The ***triple sum*** simply adds up the squares of all the individual $\Theta$s in the entire network.

-   The $i$ in the ***triple sum*** does **not** refer to training example $i$.

## Backpropagation Algorithm
"Backpropagation" is neural-network terminology for ***minimizing*** our `cost function`, just like what we were doing with gradient descent in logistic and linear regression.

Our goal is to compute:
$$min_{\Theta} J(\Theta)$$

That is, we want to ***minimize*** our cost function $J$ using an *optimal* set of parameters in theta.

We will use backpropagation to compute partial derivative of cost function.
The intuition of backpropagation is for each node we're going to compute the term: $\delta_j^{(l)}$ - "error" of node $j$ in layer $l$.

In this section we'll look at the equations we use to compute the `partial derivative` of $J(\Theta)$:
$$\frac{\partial}{\partial \Theta_{i,j}^{(l)}} J(\Theta)$$

To do so, we use the following algorithm:
![Backpropagation algorithm](/image/w5-1-1.png)
**Back propagation Algorithm**
Given training set $\{(x^{(1)}, y^{(1)}), ... , (x^{(m)}, y^{(m)})\}$

-   Set $\Delta_{i,j}^{(l)} := 0$ for all $(l, i, j)$, (hence you end up having a `matrix` full of ***zeros***)

For training example $t = (1$ to $m)$:
1.  Set $a^{(l)} := x^{(t)}$
2. Perform ***forward propagation*** to compute $a^{(l)}$ for $l = 2, 3, ..., L$
    ![Gradient computaiton](/image/w5-1-2.png)
3. Using $y^{(t)}$, compute $\delta^{(t)} = a^{(L)} - y^{(t)}$


    Where $L$ is our total number of layers and $a^{(L)}$ is the `vector of outputs` of the ***activation units*** for the *last* layer.
    So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in $y$.
    To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute $\delta^{L - 1}$, $\delta^{L - 2}$, ... ,  $\delta^{2}$ using $\delta^{l}$ = $((\Theta^{(l)})^T \delta^{(l+1)}).* a^{(l)} .* (1 - a^{(l)})$

    The delta values of layer $l$ are calculated by multiplying the delta values in the next layer with the theta matrix of layer $l$.
    We then element-wise multiply that with a function called $g'$, or g-prime, which is the ***derivative*** of the `activation function` $g$ evaluated with the input values given by $z^{(l)}$.

    The g-prime derivative terms can also be written out as:
    $$g'(z^{l}) = a^{(l)}\ .* \ (1 - a^{(l)})$$

5. $\Delta_{i,j}^{(l)} := \Delta_{i, j}^{(l)} + a_j^{(l + 1)}$
or with `vectorization`,
$\Delta^{(l)} := \Delta^{(l)} + \delta^{(l + 1)}(a^{(l)})^T$

Hence we update our new $\Delta$ matrix.

-   $D_{i,j}^{(l)} := \frac{1}{m}(\Delta_{i,j}^{(l)} + \lambda \Theta_{i,j}^{(l)})$, if $j \ne 0$.

-   $D_{i,j}^{(l)} := \frac{1}{m} \Delta_{i,j}^{(l)}$, if $i = 0$

The capital-delta matrix $D$ is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $\dfrac{\partial}{\partial \Theta_{ij}^{(l)}} J(\Theta) = D_{ij}^{(l)}$


## Backpropagation Intuition
Recall that the cost function for a neural network is:
$$J(\Theta) = -\frac{1}{m} \sum_{t = 1}^{m} \sum_{k = 1}^{K}[y_k^{(t)}log(h_\Theta(x^{(t)})_k + (1 - y_k^{(t)})log(1 - h_\Theta(x^{(t)})_k)] + \frac{\lambda}{2m}\sum_{l = 1}^{L - 1} \sum_{i = 1}^{s_l} \sum_{j = 1}^{s_l + 1}(\Theta_{j,i}^{(l)})^2  $$

If we consider simple non-multiclass classification (k = 1) and disregard regularization, the cost is computed with:
$cost(t) = y^{(t)}log(h_\Theta(x^{(t)})) + (1 - y^{(t)})log(1 - h_\Theta(x^{(t)}))$

Intuitively, $\delta_j^{(t)}$ is the "error" for $a_j^{(l)}$ (unit $j$ in layer $l$). More formally, the delta values are actually the derivative of the cost function:
$\delta_j^{(l)} = \frac{\partial}{\partial z_j^{(l)}} cost(t)$

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope, the more incorrect we are. Let us consider the following neural network below and see how we could calculate some $\delta_j^{(l)}$:
![Forward Propagatiown](/image/w5-1-3.png)

In the image above, to calculate $\delta_2^{(2)}$, we multiply the weights $\Theta_{12}^{(2)}$ and $\Theta_{22}^{(2)}$ by their repective $\delta$ values found to the right of each edge.
So we get $\delta_2^{(2)} = \Theta_{12}^{(2)} * \delta_1^{(3)} + \Theta_{22}^{(2)} * \delta_2^{(3)}$.
To calculate every single possible $\delta_j^{(l)}$, we could start from the ***right*** of our diagram.
We can think of our edges as our $\Theta_{ij}$. Going from right to left, to calculate the value of $\delta_j^{(l)}$, you can just take the over all sum of each weight times the $\delta$ it is coming from.
Hence, another example would be $\delta_2^{(3)} = \Theta_{12}^{(3)} * \delta_1^{(4)}$
