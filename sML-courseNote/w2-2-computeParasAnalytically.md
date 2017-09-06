# Normal Equation

**Gradient descent** gives one way of minimizing $J$.
Let's discuss a *second* way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm.
In the **Normal Equation** method, we will minimize $J$ by explicitly taking its derivatives with respect to the $\theta_j$'s, and setting them to `zero`. This allows is to find the *optimum* **theta** *without* **iteration**. The normal equation formula is given below:
$$\theta = (X^TX)^{-1}X^Ty$$
![](/image/w2-2-1.png)
There is **no need** to do `feature scaling` with the normal equation.

| **Gradient Descent** | **Normal Equation** |
| --- | --- |
| Need to choose alpha | No need to choose alpha |
| Needs many iterations | No need to iterate |
| $O(kn^2)$ | $O(n^3)$, need to calculate inverse of $X^TX$ |
| Works well when $n$ is large | Slow if $n$ is very large |

With the normal equation, computing the inversion has complexity $O(n^3)$. So if we have a very ***large*** `number of features`, the normal equation will be ***slow***.
-   In practice, when $n$ exceeds $10,000$ it might be a good time to go from a `normal` solution to an `iterative` process.

# Normal Equation Noninvertibility
When implementing the normal equation in octave we want to use the **`pinv`** function rather than `inv`.
-   The `pinv` function will give you a value of $\theta$ even if $X^TX$ is not invertible.

if $X^TX$ is **noninvertible**, the common causes might be having:
-   **Redundant features**, where two features are very closely related (i.e. they are linearly dependent)
-   Too many features (e.g. $m \le n$). In this case, delete some features or use "regularization" ( to be explained in a later lesson).

Solutions to the above problems include 
-   deleting a feature that is **linearly dependent** with another *or*
-   deleting one or more features when there are too **many** features.
