#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import string
from tec.exception import LexerException


class Input(object):
    def __init__(self, input):
        self.index = 0
        self.length = len(input)
        self.input = input
        # current position
        self.column = 0

    def next(self):
        if self.index < self.length:
            literal = self.input[self.index]
            self.index += 1
            self.column += 1
            return literal
        else:
            return None

# allowed character
ALPHA = list(string.ascii_letters)

# allowed functions
KEYWORD = ["sin", "cos", "exp", "log",
           "sign", "abs", # "adj", "det", "logdet", \
           # "matrix", "vector", "diag", "inv", "tr",
           "sum", "sqrt", "norm1", "norm2", "forall"]

# allowed numbers
DIGIT = [str(i) for i in range(10)]

# allowed symbols
SYMBOL = {'(': 'lrbracket',
          ')': 'rrbracket',
          '[': 'lsbracket',
          ']': 'rsbracket',
          '*': 'times',
          '+': 'plus',
          ',': 'comma',
          '-': 'minus',
          '/': 'div',
          '^': 'pow',
          #'.': 'dot',
          ":": 'colon',
          '\'': 'prime',
          '.*': 'ptimes',
          #'./': 'pdiv',
          # '=': 'equal',
          # ':=': 'assign',
          # '==': 'doubleequal',
          # '>': 'greater',
          # '<': 'lower',
          }

# allowed compare operations
# COMPARE = ["==",":=","<=","<",">=",">"]


class Scanner(object):
    def __init__(self, input):
        self.input = Input(input)
        self.symbols = []
        self.current = self.input.next()

    def get_sym(self):
        identifier = ""
        description = ""
        # skip blanks
        while self.current is " ":
            self.current = self.input.next()
        # detects number
        if self.current in DIGIT:
            description = "number"
            identifier += self.current
            self.current = self.input.next()
            while self.current in DIGIT:
                identifier += self.current
                self.current = self.input.next()
            if self.current == ".":
                identifier += self.current
                self.current = self.input.next()
                if self.current in DIGIT:
                    while self.current in DIGIT:
                        identifier += self.current
                        self.current = self.input.next()
                else:
                    raise LexerException("DIGIT expected.", self.get_column())
        # detects variables, index and functions
        elif self.current in ALPHA:
            while self.current in ALPHA:
                identifier += self.current
                self.current = self.input.next()
            # detects variables
            if identifier not in KEYWORD:
                description = "variable"
            # detects the sum function
            elif identifier == "sum":
                description = "sum"
            # detects the forall function
            elif identifier == "forall":
                description = "forall"
            # detects functions
            elif identifier in KEYWORD:
                description = "function"
            else:
                raise LexerException("Either a variable, nor a function.", self.get_column())
        # detects symbols
        elif self.current in SYMBOL.keys():
            description = SYMBOL.get(self.current)
            identifier = self.current
            self.current = self.input.next()
        # detects if the scan is finished
        elif self.current is None:
            description = "none"
            identifier = self.current
        # unrecognizes symbol
        else:
            description = "error"
            identifier = self.current
            raise LexerException("Symbol '%s' not allowed" % identifier, self.get_column())

        return (description, identifier)

    # get the column number for error tracing
    def get_column(self):
        return self.input.column
