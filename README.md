# TEC: Tensor Expression Compiler

Transform textbook like expressions into highly-tuned Python library code.

## Requirements
To verify our transformation we numerically evaluate the expression with NumPy and generated for-loops.
Therefore, `NumPy` is needed. If you do not want this, you can turn the verfication of.

## Experiments
The repository contains a comparison of evaluating the objective function of the Tucker decomposition using for-loops and Numpy, TensorFlow and PyTorch.
To run the comparision. Install the named frameworks and then run:

```
python discussion.py
```
