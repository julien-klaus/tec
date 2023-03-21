# TEC: Tensor Expression Compiler

Transform textbook like expressions into highly-tuned Python library code.

## Requirements
To verify our transformation we numerically evaluate the expression with NumPy and generated for-loops. You can install NumPy using
```
$ pip install numpy
``` 

## Usage
TEC is written as command line tool. You can translate an expression using
```
$ python main.py "sum[i](x[i]*A[i,j])
```

See the documention for the parameter description
```
$ python main.py -h
usage: main.py [-h] [-l {numpy,tensorflow,pytorch}] [-v] expression

Translate textbook formulas into Einsum.

positional arguments:
  expression            textbook formula for the translation, e.g. sum[i](x[i]*A[i,j])

optional arguments:
  -h, --help            show this help message and exit
  -l {numpy,tensorflow,pytorch}, --library {numpy,tensorflow,pytorch}
                        translate into numpy (default), pytorch or tensorflow
  -v, --verbose         shows the intermediate program code for verifying the translation
```

## Experiments
The repository contains a comparison of evaluating the objective function of the Tucker decomposition using for-loops and Numpy, TensorFlow and PyTorch.

You can install them using
```
$ pip install numpy torch tensorflow
```

After the installation you can run the experiment

```
$ python discussion.py
Comparision of evaluating the objective function of the tucker decomposition for a tensor of size (s,s,s).

Start experiment with s = 25
        * Baseline computed in 3.05585s
        * NumPy computed in 0.01376s
        * PyTorch computed in 0.00050s
        * TensorFlow computed in 0.09407s
        -> Speed-Ups for s = 25 (Baseline Python, NumPy, TensorFlow, PyTorch): 1, 222, 32, 6146
...
```
