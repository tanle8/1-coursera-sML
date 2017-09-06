# Building a Spa Classifier

# Handling Skewed Data
### Terminology
-   `Norm`
    A norm is a function that defines some notion of length.
    Specifically, a norm is a function that maps an input to a strictly positive number, other than a special input called the zero input.
    Norms must also satisfy linearity properties.
    The most common example of a norm is the Euclidean norm which is our typical notion of the length of a vector.


-   `Precision`
    Precision (also called positive predictive value) is the fraction of examples predictive to be positive that are actually positive.
    In other words, precision is the number of `true positives` divided by number of `total positives` **predicted by the model**.
    Precision can be thought of as the probability that an example predicted to be positive will actually be positive.


-   `Recall`
    Recall (also known as sensitivity) is the fraction of positive examples that the model identifies to be positive.
    Recall can be thought of as the probability that a `positive example` is **correctly recognized** to be `positive` by the `model`.


### Error Metrics for Skewed Classes
Precision and Recall are defined according to:
![Precision and Recall](/image/w6-2-2.png)

$\text{Precision} = \frac{\text{True positives}}{\text{# predicted as positive}} = \frac{\text{True positives}}{\text{True positives + False positives}}$

$\text{Recall} = \frac{\text{True positives}}{\text{# actual positives}} = \frac{\text{True positives}}{\text{True positives + False negatives}}$


### Trading Off Precision and Recall
For example, in logistic regression: $0 \le h_\theta(x) \le 1$
-   Predict 1 if $h_\theta(x) \ge threshold$
-   Predict 0 if $h_\theta(x) < threshold$

Suppose we want to predict $y = 1$ (cancer) only if very confident.
        --> Higher precision, Lower recall.

Suppose we want to avoid missing too many cases of cancer (avoid false negatives).
        --> Higher recall, Lower precision.
More generally: Predict $1$ if $h_\theta(x) \ge threshold$

#### F1 Score (F score)
How to compare precision/recall numbers?

Average $\frac{P + R}{2}$ is **not** good enough, in the case when your model predict y = 1 (or y = 0) all the time.

Usually, we use $F_1 score: 2\frac{PR}{P + R}$.
This gives us a single real number evaluation metric.

And if our goal is to automatically set that threshold to decide what is really $y = 1$ and $y = 0$, one reasonable way to do is to try a range of different values of thresholds on the cross-validation set and then to pick whatever value of threshold gives us the highest $F score$ on our `cross-validation set`.

# Using Large Data Sets
