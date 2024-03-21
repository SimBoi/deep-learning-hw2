r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. The Jacobian tensor $\frac{\delta Y}{\delta X}$ of the output of the layer w.r.t. the input $X$.

1.A. The shape of this tensor is (N, out_features, N, in_features) = (64, 512, 64, 1024). This is because for each of the N samples, we have a gradient of a vector of length out_features w.r.t a vector of length in_features.

1.B. This Jacobian is not sparse. In a fully connected layer, each output neuron is connected to each input neuron, so the derivative of each output w.r.t. each input is generally non-zero.

1.C. We do not need to materialize the Jacobian to calculate the downstream gradient w.r.t. to the input. This is because the backpropagation algorithm allows us to compute the product of the Jacobian with the upstream gradient without explicitly forming the Jacobian itself. The downstream gradient $\frac{\delta L}{\delta X}$ can be calculated as $\frac{\delta L}{\delta Y}\cdot W^T$.



2. The Jacobian tensor $\frac{\delta W}{\delta X}$ of the output of the layer w.r.t. the layer weights $W$.

2.A. The shape of this tensor is (N, out_features, out_features, in_features) = (64, 512, 512, 1024). This is because for each of the N samples, we have a gradient of a vector of length out_features w.r.t a matrix of size (out_features, in_features).

2.B. This Jacobian is not sparse. In a fully connected layer, each output neuron is connected to each weight, so the derivative of each output w.r.t. each weight is generally non-zero.

2.C. Similar to 1.C, we do not need to materialize the Jacobian to calculate the downstream gradient w.r.t. to the weights. The downstream gradient $\frac{\delta L}{\delta W}$ can be calculated as $X^T \cdot\frac{\delta L}{\delta Y}$.

"""

part1_q2 = r"""
**Your answer:**

Yes, back-propagation is necessary for training neural networks with gradient-based optimization. It calculates the gradients needed to update the network’s weights to minimize the loss. Without it, we wouldn’t have an efficient way to compute these gradients.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. The obtained graphs indicate that the model without dropout outperforms the models with dropout in terms of training set accuracy. However, in the test set, the model with the highest dropout has the lowest accuracy, while the no-dropout and low-dropout models have nearly identical accuracies. This contradicts our initial expectations, as we anticipated the no-dropout model to overfit and thus have lower accuracies. We suspect that the issue might lie in the hyperparameters' values. Adjusting these might lead to the anticipated results.
2. The graphs demonstrate that a lower dropout rate (0.4) yields higher accuracies in both the training and test sets compared to a higher dropout rate (0.8). This aligns with our expectations, as a moderate dropout rate can prevent overfitting while still utilizing a significant number of neurons. A very high dropout rate (0.8) can hinder performance, as the model relies less frequently on the neurons, potentially leading to underfitting.

"""

part2_q2 = r"""
**Your answer:**

Yes, it’s possible. The cross-entropy loss takes into account the certainty of the model’s predictions. Therefore, even if the accuracy improves (due to more predictions being correct), the loss might still rise if the model’s certainty in its predictions diminishes. For instance, in a binary classification scenario of 0/1, if the actual label is 1 and the model predicts 0.51, it will be rounded up to 1, thus improving accuracy. However, the loss will be significant due to the model’s low confidence in this prediction. Situations like outliers and imbalanced classes can contribute to this phenomenon.

"""

part2_q3 = r"""
**Your answer:**

1. Gradient descent is an optimization algorithm used to minimize a function (like loss function) by iteratively moving in the direction of steepest descent. Back-propagation is a method used to calculate the gradient of the loss function with respect to the weights in the network.

2. GD uses all data points to compute the gradient, while SGD uses a single or a few samples at each iteration. This makes SGD faster and able to handle large datasets.

3. SGD is more commonly used in deep learning because it's computationally efficient, handles redundant data well, and introduces noise into the learning process, which can prevent overfitting.

4.
   1. No, this approach wouldn't produce a gradient equivalent to GD. GD requires the gradient to be calculated over the entire dataset, not batch by batch.
   2. The out of memory error might have occurred because the sum of the losses from all batches was stored in memory, which can be large for big datasets.

"""

part2_q4 = r"""
**Your answer:**

1.
    1. In forward mode Automatic Differentiation (AD), we can compute the gradient one component at a time, which reduces the memory complexity to $\mathcal{O}(1)$.
    2. In backward mode AD, we can use a technique called "checkpointing" to reduce the memory complexity. We save certain intermediate values during the forward pass and recompute the others during the backward pass. This reduces the memory complexity to $\mathcal{O}(n)$.

2. Yes, these techniques can be generalized for arbitrary computational graphs.

3. Backpropagation, used in deep learning, can benefit from these techniques by reducing the memory usage, especially in deep architectures like VGGs and ResNets where memory can be a bottleneck. Checkpointing can be particularly useful in these cases.

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

1. The Optimization error doesn't appear to be high, as the training set loss decreases steadily and stabilizes at a constant value.
2. The Generalization error doesn't seem to be high either. Despite some fluctuations in the graph, the model achieves good accuracy, indicating that while there's room for improvement, the error isn't excessively high.
3. The Approximation error is not high, given that the model correctly classifies the majority of the samples.

"""

part3_q2 = r"""
**Your answer:**

Based on the graph, it appears that there are more red dots misclassified as positive than blue dots misclassified as negative. Therefore, we can infer that the False Positive Rate (FPR) is likely to be higher than the False Negative Rate (FNR).

"""

part3_q3 = r"""
**Your answer:**

The "optimal" point on the ROC curve would depend on the cost and risk associated with false positives and false negatives, which varies in both scenarios:

1. If a person with the disease will develop non-lethal symptoms that confirm the diagnosis, we might want to be more conservative with our positive predictions to avoid unnecessary high-risk tests. This means we might choose a point on the ROC curve that favors specificity (true negative rate) over sensitivity (true positive rate).

2. If a person with the disease shows no clear symptoms and may die if not diagnosed early, we would want to catch as many true positives as possible, even at the risk of more false positives. This means we might choose a point on the ROC curve that favors sensitivity over specificity.

In both cases, the choice of the "optimal" point would be a trade-off between sensitivity and specificity, taking into account the costs and risks associated with false positives and false negatives.

"""


part3_q4 = r"""
**Your answer:**

1. Observing the columns where `depth` is fixed and `width` varies, it appears that a width of 32 yields the best results in the first two columns. However, in the last column, a width of 8 seems to be optimal.

2. Looking at the rows where `width` is fixed and `depth` varies, it seems that a depth of 2 is best in the first row, a depth of 4 in the second, and a depth of 1 in the last row.

3. Comparing the two configurations with the same total number of parameters, `depth=1, width=32` and `depth=4, width=8`, both perform well. However, the configuration with `depth=1` and `width=32` seems to perform better. This is contrary to our expectation that a more balanced configuration would perform better, suggesting that our choice of hyperparameters may have influenced the outcome.

4. The selection of the threshold on the validation set did influence the results on the test set. By adjusting the threshold, we control the trade-off between precision and recall, which lead to better performance on the test set.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

Sure, here are the formulas written using LaTeX:

1. If we include a bias parameter, the number of parameters for each layer can be calculated using the formula:
$$\text{{parameters}} = (\text{{kernel size}} \times \text{{input channels}} + 1) \times \text{{output channels}}$$
For the regular block, this results in
$$(3 \times 3 \times 256 + 1) \times 256 \times 2 = 1,180,160 \text{{ parameters}}$$
For the bottleneck, the calculation is
$$(256 \times 1 \times 1 + 1) \times 64 + (64 \times 3 \times 3 + 1) \times 64 + (64 \times 1 \times 1 + 1) \times 256 = 70,016 \text{{ parameters}}$$

2. The number of floating point operations (FLOPs) for the regular block is
$$(3 \times 3 \times 256) \times 256 \times 2 = 1,179,648$$
while for the bottleneck it's
$$(256 \times 1 \times 1) \times 64 + (64 \times 3 \times 3) \times 64 + (64 \times 1 \times 1) \times 256 = 69,632$$

3. The regular block has a larger receptive field, making it better for spatial combination within feature maps. However, the bottleneck, with its deeper network and more layers, has an advantage when it comes to combining across feature maps.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. The model did not detect the objects that well, we can see that in multiple ways:
    - In the dogs and cat image the bounding boxes were too large compared to the background, and it caused them to overlap which gave a skewed classification on the dogs, and completely missed and object.
    - The labels given in the dolphins completely missed the mark.

2. There can be multiple reasons for the bad results, in the case of the cat and dogs, the cat was blocking a large part of the Shiba-Inus which can confuse the model of where the anchoring box should be and how big.
    In the dolphins image, there was also overlap between the dolphins where one blocked the other, in addition to the bad illumination because of the camera facing the sun which took away valuable information like texture and color, there was also an apparent bias towards the background and environment, dolphins are not expected to be outside of water, while a scenary like can be associated with people surfing.
    We can maybe solve this by changing the bias of pixels outside the the bouding boxes, and train the model on more images with occlusion.
"""


part6_q3 = r"""
**Your answer:**
We managed to get even worse results due to a different reason for each image:
1.  In the `Blurry.jpg` image, the model failed to even detect an object at all, and assumed the moving cup was part of the background.
2.  Int the `Occluded.jpg` image, it was able to detect a chair but the bounding box made no sense and was too small, it also managed to detect a weird tie as part of the chair.
3.  In `Dark.jpg` it mislabeled the wolf as a horse due to lack of illumination, similar to the dolphins photo, the background bias might've also played a role.
"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""