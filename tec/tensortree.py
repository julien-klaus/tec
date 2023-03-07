#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import string
from copy import deepcopy
import os

from tec.exception import SemanticException


class TensorTree(object):
    def __init__(self, name, left=None, right=None, type=None):
        self.name = name
        self.left = left
        self.right = right
        self.type = type
        self.index_to_shape = {}
        self.index = None

    def copy(self):
        return deepcopy(self)

    def pp(self, lvl=1):
        ret = "|" + "-" * lvl + ": " + str(self.name) + " (" + self.type + ")" + f" - (index: {self.index})\n"
        if self.left is None:
            ret += ""
        else:
            ret += self.left.pp(lvl + 2)
        if self.right is None:
            ret += ""
        else:
            ret += self.right.pp(lvl + 2)
        return ret

    def __str__(self):
        return self.pp()
    
    def __repr__(self):
        return self.pp()

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_scope(self):
        return self.scope

    def get_type(self):
        return self.type

    def get_number_of_nodes(self):
        return len(self.get_nodes())

    def get_variables(self):
        variables = dict()
        for node in self.get_nodes():
            if node.get_type() == "variable":
                temp = dict()
                if node.left is not None and node.left.get_type() == "index":
                    temp["left"] = node.left.get_name()
                # save object
                temp["vectree"] = node
                if node.get_name() in variables and len(temp) != len(variables[node.get_name()]):
                    raise SemanticException("Variable %s is defined with differents types." % node.get_name())
                # save properties
                variables[node.get_name()] = temp
        return variables

    def is_scalar(self):
        return self.type == "variable" and not self.left

    def is_index(self):
        return self.type == "index" and self.left is None and self.right is None

    def is_vector(self):
        return self.type == "variable" and self.left and len(self.left) == 1 and self.left.type == "index"

    def is_matrix(self):
        return self.type == "variable" and self.left and len(self.left) == 2 and self.left.type == "index"

    def is_tensor(self):
        return self.type == "variable" and self.left and len(self.left) > 2 and self.left.type == "index"

    def get_nodes(self):
        nodes = []
        if self is not None:
            nodes.append(self)
        if self.left is not None:
            nodes.extend(self.left.get_nodes())
        if self.right is not None:
            nodes.extend(self.right.get_nodes())
        return nodes

    def set_index_property(self):
        # recursivly create the python code
        if self.get_type() == "index":
            self.index = self.get_name()
        elif self.get_type() == "variable":
            if self.left:
                self.left.set_index_property()
                self.index = self.left.index
        elif self.get_type() == "scalar":
            self.index = []
        elif self.get_type() == "function":
            self.left.set_index_property()            
            self.index = self.left.index
        elif self.get_type() == "sum":
            self.left.set_index_property()
            self.right.set_index_property()
            sum_index = self.right.index.copy()
            # remove the summation indices
            for index in self.left.index:
                sum_index.remove(index)
            self.index = sum_index
        elif self.get_type() == "operation":
            self.left.set_index_property()
            union_index = self.left.index.copy()
            if not self.right is None:
                self.right.set_index_property()
                # combine left and right
                for index in self.right.index:
                    if not index in union_index:
                        union_index.append(index)
            self.index = union_index
        else:
            raise Exception(f"Could not compute indices. Node type '{self.get_type()}' not known.")
        
    def flat_sum(self):
        """
        Flat the sum node in such a way, that every index has a seperate sum node. E.g. sum[i,j](..) = sum[i](sum[j](..)).
        """
        if self.get_type() == "sum":
            tmp = self
            first_index = self.left.get_name()[0]
            running_index_list = self.index.copy()
            running_index_list.append(first_index)
            for index in self.left.get_name()[1:]:
                index_node = TensorTree([index], type="index")
                index_node.index = [index]
                tt = TensorTree(self.get_name(), index_node, tmp.right, "sum")
                tmp.right = tt
                tt.index = running_index_list.copy()
                running_index_list.append(index)
                tmp = tt 
            self.left = TensorTree([first_index], type="index")
            self.left.index = [first_index]
            tmp.right.flat_sum()
        else:
            if self.left:
                self.left.flat_sum()
            if self.right:
                self.right.flat_sum()

    def set_shapes(self):
        if self.get_type() == "variable":
            for i, index in enumerate(self.index):
                self.index_to_shape[index] = f"{self.get_name()}.shape[{i}]"
        else:
            if self.left:
                self.left.set_shapes()
                self.index_to_shape.update(self.left.index_to_shape.copy())
            if self.right:
                self.right.set_shapes()
                self.index_to_shape.update(self.right.index_to_shape.copy())
    
    #TODO: Double indices should be allowed, refactor this
    def check_indices(self):
        if self.get_type() in ["sum", "variable"]:
            if len(set(self.left.get_name())) != len(self.left.get_name()):
                raise Exception("Double indices are not allowed. Please reformulate your problem.")
            if self.get_type() == "sum":
                self.right.check_indices()
        else:
            if self.left:
                self.left.check_indices()
            if self.right:
                self.right.check_indices()
