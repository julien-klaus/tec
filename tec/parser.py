#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tec.lexer import Scanner
from tec.tensortree import TensorTree
from tec.exception import ParserException


"""
expr ::= term ( ( '+' | '-' ) term )*
term ::= factor ( ( '*' | '\' ) factor )*
factor ::= atom ( '^' ('(' factor ')' | atom ))*
atom ::= number | function '(' expr ')' | variable
number ::= ["-"] digit* '.' digit*
digit ::= [0-9]
functon ::= 'sin' | 'cos' | 'exp' | 'log' | 'sqrt' | 'abs' | 'sum' '[' index ']'
index ::= alpha
alpha ::= [a-zA-Z]
variable ::= alpha+ ( '[' indices ']' )?
indices ::= alpha {',' alpha}
"""


class Parser(object):
    def __init__(self,input):
        self.indices = dict()
        self.scanner = Scanner(input)
        self.get_sym()
        self.tensortree = self.expr()
        self.tensortree.indices = self.indices
        self.tensortree.calculate_scopes()
        if self.identifier is not None:
            raise ParserException("Expected nothing got '%s'." % self.identifier, self.scanner.get_column())

    def expr(self):
        tensortree = self.term()
        while self.description_equals("plus") or self.description_equals("minus"):
            ident = self.identifier
            self.get_sym()
            tensortree = TensorTree(name=ident, left=tensortree, right=self.term(), type="operation")
        return tensortree

    def term(self):
        tensortree = self.factor()
        while self.description_equals("times") or self.description_equals("div"):
            ident = self.identifier
            self.get_sym()
            tensortree = TensorTree(name=ident, left=tensortree, right=self.factor(), type="operation")
        return tensortree

    def factor(self):
        desc = self.description
        ident = self.identifier
        if self.description_equals("minus"):
            self.get_sym()
        if desc  == "minus":
            tensortree = TensorTree(ident, left=self.atom(), type="operation")
        else:
            tensortree = self.atom()
        while self.description_equals("pow"):
            ident = self.identifier
            self.get_sym()
            if self.description_equals("lrbracket"):
                self.get_sym()
                tensortree = TensorTree(name=ident, left=tensortree, right=self.factor(), type="operation")
                if self.description_equals("rrbracket"):
                    self.get_sym()
                else:
                    raise ParserException("Missing ')'.", self.scanner.get_column())
            else:
                tensortree = TensorTree(name=ident, left=tensortree, right=self.atom(), type="operation")
        return tensortree

    def atom(self):
        ident = self.identifier
        tensortree = TensorTree(name="error")
        if self.description_equals("function"):
            self.get_sym()
            if self.description_equals("lrbracket"):
                self.get_sym()
                tensortree = TensorTree(name=ident, left=self.expr(), type="function")
                if self.description_equals("rrbracket"):
                    self.get_sym()
                else:
                    raise ParserException("Missing ')'.", self.scanner.get_column())
            else:
                raise ParserException("Missing '('.", self.scanner.get_column())
        elif self.description_equals("sum"):
            self.get_sym()
            if self.description_equals("lsbracket"):
                index_set = self.index()
                if self.description_equals("rsbracket"):
                    self.get_sym()
                    if self.description_equals("lrbracket"):
                        self.get_sym()
                        # For each index in 'index_set' create one sum node
                        indices = index_set.get_name()
                        root = TensorTree("sum", left=TensorTree(name=[indices[0]],type="index"), type="sum")
                        tensortree = root
                        for index in indices[1:]:
                            tensortree.right = TensorTree("sum", left=TensorTree(name=[index],type="index"), type="sum")
                            tensortree = tensortree.right
                        tensortree.right = self.expr()
                        tensortree = root
                        if self.description_equals("rrbracket"):
                            self.get_sym()
                        else:
                            raise ParserException("Missing ')'.", self.scanner.get_column())
                    else:
                        raise ParserException("Missing ')'.", self.scanner.get_column())
                else:
                    raise ParserException("Missing ].", self.scanner.get_column())
            else:
                raise ParserException("Missing index.", self.scanner.get_column())
        elif self.description_equals("number"):
            tensortree = TensorTree(name=ident, type="scalar")
            self.get_sym()
        elif self.description_equals("variable"):
            self.get_sym()
            if self.description_equals("lsbracket"):
                tensortree = TensorTree(name=ident, left=self.index(), type="variable")
                if self.description_equals("rsbracket"):
                    self.get_sym()
                else:
                    raise ParserException("Missing ']'.", self.scanner.get_column())
            else:
                tensortree = TensorTree(name=ident, type="variable")

        elif self.description_equals("lrbracket"):
            self.get_sym()
            tensortree = self.expr()
            if self.description_equals("rrbracket"):
                self.get_sym()
            else:
                raise ParserException("Missing ')'.", self.scanner.get_column())
        else:
            raise ParserException("Missing atom.", self.scanner.get_column())
        return tensortree

    def index(self):
        index_set = []
        if self.description_equals("lsbracket"):
            self.get_sym()
            if self.description_equals("variable"):
                # index should only be one letter
                if len(self.identifier) > 1:
                    raise ParserException("Index should contain only one letter.", self.scanner.get_column())
                # give each index a number for later use
                if not self.identifier in self.indices:
                    self.indices[self.identifier] = len(self.indices)
                index_set.append(self.identifier)
                # continue if index is only one letter
                self.get_sym()
                while self.description_equals("comma"):
                    self.get_sym()
                    if self.description_equals("variable"):
                        if len(self.identifier) > 1:
                            raise ParserException("Index should contain only one letter.", self.scanner.get_column())
                        # give each index a number for later use
                        if not self.identifier in self.indices:
                            self.indices[self.identifier] = len(self.indices)
                        index_set.append(self.identifier)
                        self.get_sym()
                    else:
                        raise ParserException("Index identifier expected.", self.scanner.get_column())
            else:
                raise ParserException("Index identifier expected.", self.scanner.get_column())
            if not self.description_equals('rsbracket'):
                raise ParserException("Missing ']'", self.scanner.get_column())
        # tensortree containing a list of all indices
        tensortree = TensorTree(name=index_set, type="index")
        return tensortree

    def description_equals(self, symbol):
        return self.description == symbol

    def get_sym(self):
         (self.description, self.identifier) = self.scanner.get_sym()

    def get_tensortree(self):
        return self.tensortree

