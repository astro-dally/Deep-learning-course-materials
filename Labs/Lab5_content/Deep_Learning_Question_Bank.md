# Deep Learning Question Bank (Advanced)

> **Context**: Preparation for Contest 1 | **Focus**: Numericals, Theory, Application

This document contains a curated list of high-complexity questions. Answers are hidden to facilitate active recall/self-testing.

## Table of Contents
- [Vectors & Tensors](#vectors--tensors)
- [Matrix Operations](#matrix-operations)
- [Norms](#norms)
- [Eigenvectors and Eigenvalues](#eigenvectors-and-eigenvalues)
- [Eigen & Singular Value Decomposition](#eigen--singular-value-decomposition)
- [Probability Distributions](#probability-distributions)
- [Bayes Theorem & Chain Rule](#bayes-theorem--chain-rule)
- [Information Theory](#information-theory)
- [XOR Problem](#xor-problem)
- [Forward Propagation](#forward-propagation)
- [Activation Functions](#activation-functions)
- [Shallow vs. Deep Networks](#shallow-vs-deep-networks)
- [Universal Approximation Theorem](#universal-approximation-theorem)

---

## Vectors & Tensors

### Q1. [Numerical]
**Problem**:
You have a batch of 10 color images, each 32x32 pixels. The tensor shape is (10, 3, 32, 32). How many total floating point numbers are in this tensor?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $10 \times 3 \times 32 \times 32 = 10 \times 3 \times 1024 = 30,720$ elements.

</details>

---

### Q2. [Theory]
**Problem**:
What is the difference between a scalar, a vector, and a matrix in terms of 'Rank'?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Scalar is Rank-0 (magnitude only). Vector is Rank-1 (magnitude and direction). Matrix is Rank-2 (table of numbers/transform). Tensor is Rank-N (generalization).

</details>

---

### Q3. [Application]
**Problem**:
Why do we flatten an image tensor before feeding it into a basic Fully Connected Network?

<details>

<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Fully Connected layers expect a 1D feature vector for each sample. They process global combinations of inputs and cannot inherently handle 2D spatial grid structures.

</details>

---

## Matrix Operations

### Q1. [Numerical]
**Problem**:
Given $A = [[1, 2], [3, 4]]$ and $x = [1, 0]^T$. Calculate $Ax$.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $[[1*1 + 2*0], [3*1 + 4*0]] = [1, 3]^T$.

</details>

---

### Q2. [Theory]
**Problem**:
Is matrix multiplication commutative ($AB = BA$)? Give a simple counter-example if not.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> No. Let $A$ be a projection onto X-axis and $B$ be a 90-degree rotation. Projecting then Rotating is different from Rotating (which moves X to Y) then Projecting (which kills Y).

</details>

---

### Q3. [Application]
**Problem**:
In the equation $y = Wx + b$, what is the effect of $b$ (bias) strictly in geometric terms?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> It shifts the resulting transformed vector away from the origin. Without bias, any decision boundary (hyperplane) must pass through the origin $(0,0)$.

</details>

---

## Norms

### Q1. [Numerical]
**Problem**:
Calculate the L1 norm and Squared L2 norm of the vector $v = [-14, 31]$.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> L1 = $|-14| + |31| = 45$. Squared L2 = $(-14)^2 + 31^2 = 196 + 961 = 1157$.

</details>

---

### Q2. [Theory]
**Problem**:
Which norm is equivalent to 'Euclidean Distance'?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> The L2 Norm.

</details>

---

### Q3. [Application]
**Problem**:
If you want to drive some weights in your model to exactly zero (feature selection), which regularization should you use: L1 or L2?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> L1 Regularization (Lasso), because its gradient is constant (shifts weights by a fixed amount towards zero), allowing them to hit zero exactly.

</details>

---

## Eigenvectors and Eigenvalues

### Q1. [Numerical]
**Problem**:
If vector $v$ is an eigenvector of matrix $A$ with eigenvalue $\lambda=3$, what is $A v$? What is $A (2v)$?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $A v = 3v$. Since linear, $A(2v) = 2(A v) = 2(3v) = 6v$.

</details>

---

### Q2. [Theory]
**Problem**:
What does a negative eigenvalue signify geometrically?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> It indicates that the transformation reverses the direction of the eigenvector (flips it) and then scales it.

</details>

---

### Q3. [Application]
**Problem**:
Why are eigenvalues relevant to the stability of a network's training?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> If eigenvalues of the weight matrices are very large ($>1$), activations/gradients can grow exponentially (explode) as they pass through many layers. If small ($<1$), they vanish.

</details>

---

## Eigen & Singular Value Decomposition

### Q1. [Numerical]
**Problem**:
An image has singular values $[50, 4, 1, 0.1]$. Identify the 'Signal' vs 'Noise' components.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Signal: 50 (Primary structure) and maybe 4. Noise: 1 and 0.1 (negligible detail).

</details>

---

### Q2. [Theory]
**Problem**:
Can every matrix be decomposed using SVD?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Yes, SVD exists for any rectangular matrix (unlike Eigendecomposition which strictly requires square matrices).

</details>

---

### Q3. [Application]
**Problem**:
How does keeping only the top-k singular values compress an image?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> It approximates the original matrix using only the most important patterns (basis images), discarding the data required to store the fine-grained/noisy details.

</details>

---

## Probability Distributions

### Q1. [Numerical]
**Problem**:
If a fair coin is tossed 3 times, what is the probability of getting exactly 3 Heads?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $0.5 \times 0.5 \times 0.5 = 0.125$ (or $1/8$).

</details>

---

### Q2. [Theory]
**Problem**:
What is the area under the curve of any Probability Density Function (PDF)?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> The total area must always sum to exactly 1.

</details>

---

### Q3. [Application]
**Problem**:
Why do we often work with Log-Probabilities instead of raw probabilities in neural networks?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> To avoid numerical underflow. Multiplying many small probabilities results in a number so close to zero the computer treats it as zero. Adding logs keeps numbers in a manageable range.

</details>

---

## Bayes Theorem & Chain Rule

### Q1. [Numerical]
**Problem**:
Given $P(A) = 0.5, P(B|A) = 1.0, P(B) = 0.8$. Calculate $P(A|B)$.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{1.0 \times 0.5}{0.8} = \frac{0.5}{0.8} = 0.625$.

</details>

---

### Q2. [Theory]
**Problem**:
What is the 'conditional independence' assumption in Naive Bayes?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> It assumes that all features are independent of each other given the class label (e.g., probability of word 'Free' does not depend on word 'Money', only on 'Spam').

</details>

---

### Q3. [Application]
**Problem**:
If a test for a disease is positive, why is the probability of having the disease often lower than the test's accuracy?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Because if the disease is very rare (low prior), the number of false positives can outnumber the true positives.

</details>

---

## Information Theory

### Q1. [Numerical]
**Problem**:
Event A has probability $0.25$. Calculate its Information (Surprisal) in bits.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $I(A) = -\log_2(0.25) = -\log_2(2^{-2}) = 2$ bits.

</details>

---

### Q2. [Theory]
**Problem**:
Which distribution has higher entropy: A coin with $P(H)=0.5$ or a coin with $P(H)=0.9$?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> The fair coin ($P=0.5$) has maximum entropy because it is the most unpredictable.

</details>

---

### Q3. [Application]
**Problem**:
Intuitively, what does 'Cross-Entropy' measure between a predicted distribution and a true label?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> It measures the 'distance' or 'error' between the prediction and the truth. If the prediction is confident and wrong, Cross-Entropy is high.

</details>

---

## XOR Problem

### Q1. [Numerical]
**Problem**:
The XOR inputs are $(0,0), (0,1), (1,0), (1,1)$ with labels $0, 1, 1, 0$. Plot these on a 2D graph. Can you draw one straight line to separate 0s from 1s?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> No. The points $(0,0)$ and $(1,1)$ are diagonally opposite, as are $(0,1)$ and $(1,0)$. Any line separating one pair leaves the other pair on the same side or mixed.

</details>

---

### Q2. [Theory]
**Problem**:
What does a hidden layer do that allows a neural net to solve XOR?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> It transforms the coordinates of the inputs into a new space where the points are rearranged so they *can* be separated by a straight line (linear classifier).

</details>

---

### Q3. [Application]
**Problem**:
Is XOR linearly separable?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> No, it is the classic example of a non-linearly separable problem.

</details>

---

## Forward Propagation

### Q1. [Numerical]
**Problem**:
Input $x=2$, weight $w=3$, bias $b=-1$. Activation is Identity. Calculate output $y$.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $y = w \cdot x + b = 3 \cdot 2 + (-1) = 6 - 1 = 5$.

</details>

---

### Q2. [Theory]
**Problem**:
Why do we organize neurons into 'layers'?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> It allows us to build complex functions through step-by-step composition. Each layer processes the features extracted by the previous layer.

</details>

---

### Q3. [Application]
**Problem**:
If you multiply the signal by 1.5 at each layer, what happens after 100 layers?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> The signal grows by a factor of $1.5^{100}$. This leads to exploding values.

</details>

---

## Activation Functions

### Q1. [Numerical]
**Problem**:
ReLU function is $f(x) = max(0, x)$. Evaluate ReLU at $x = -5, 0, 5$.

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> $ReLU(-5) = 0$. $ReLU(0) = 0$. $ReLU(5) = 5$.

</details>

---

### Q2. [Theory]
**Problem**:
Why is 'Linear' not a good activation function for deep networks?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> A stack of linear layers is mathematically equal to just one single linear layer. You lose the benefit of depth.

</details>

---

### Q3. [Application]
**Problem**:
Why is ReLU preferred over Sigmoid for deep networks?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Sigmoid gradients are small ($<0.25$) causing signals to vanish in deep nets. ReLU gradients are 1 (for positive input), allowing signals to flow without decaying.

</details>

---

## Shallow vs. Deep Networks

### Q1. [Theory]
**Problem**:
What is the main advantage of a Deep Network over a Shallow (Wide) Network?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Efficiency. Deep networks can represent complex patterns using far fewer parameters by reusing features hierarchically.

</details>

---

### Q2. [Application]
**Problem**:
Which architecture is better at learning a hierarchy of concepts (e.g., edges -> shapes -> faces)?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> A Deep Network.

</details>

---

## Universal Approximation Theorem

### Q1. [Theory]
**Problem**:
What does the Universal Approximation Theorem state simply?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> That a neural network with just one hidden layer (if wide enough) can approximate broadly any function you want.

</details>

---

### Q2. [Application]
**Problem**:
If one hidden layer is enough 'in theory', why do we use deep networks 'in practice'?

<details>
<summary><strong>Check Solution</strong></summary>

> **Answer**:
> Because the 'wide enough' layer might need to be impossibly wide (billions of neurons) to work, whereas a deep network could do it with much less.

</details>

---
