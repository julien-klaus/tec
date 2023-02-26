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
        self.tensor = None
        self.scope = set()
        self.type = type
        self.indices = None

    def copy(self):
        return deepcopy(self)

    def pp(self, lvl=1):
        ret = "|" + "-" * lvl + ": " + str(self.name) + " (" + self.type + ")\n"
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

    def check_indices(self):
        # TODO: check the bound and unbound indices (sum is not allowed to use a not used indice,
        # may also do it in the calculate scope function
        pass

    def get_unbound_indices(self):
        return self.scope

    def calculate_scopes(self):
        self._calculate_scope()

    def _calculate_scope(self):
        def _join_scope(left, right):
            if left:
                if right:
                    return left.scope.union(right.scope)
                else:
                    return left.scope
            else:
                return set()
        if self.left:
            self.left._calculate_scope()
        if self.right:
            self.right._calculate_scope()
        if self.get_type() == "index":
            self.scope = set(self.get_name())
        elif self.get_name() == "sum":
            indices = set(self.left.get_name())
            # all sum indices should be used by the variables in the subtree starting at the sum
            index_check = indices.difference(self.right.scope)
            if len(index_check) > 0:
                raise SemanticException(f"The indices {{{','.join(list(index_check))}}} "
                                        f"are not used for variables. Please check the sum indices.")
            self.scope = _join_scope(self.left, self.right)
            self.scope = self.scope - indices
        else:
            self.scope = _join_scope(self.left, self.right)

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_scope(self):
        return self.scope

    def get_type(self):
        return self.type

    def is_transformable(self):
        return self.left is not None and self.left.tensor is not None and \
               self.right is not None and self.right.tensor is not None

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

    def generate_vectree(self, file_name=os.path.join("img","vectree")):
        dot = Digraph(format="pdf")
        nodes_to_number = {}
        i = 0
        for node in self.get_nodes():
            nodes_to_number[node] = str(i)
            name = f"{node.get_name() if not isinstance(node.get_name(), list) else ','.join(node.get_name())}" \
                    + f" ({','.join(list(node.get_scope()))})"  + f" ({node.get_type()})"
            dot.node(str(i), name)
            i += 1
        for node in self.get_nodes():
            if node.left is not None:
                dot.edge(nodes_to_number[node], nodes_to_number[node.left])
            if node.right is not None:
                dot.edge(nodes_to_number[node], nodes_to_number[node.right])
        dot.render(file_name)
        return dot

    def get_nodes(self):
        nodes = []
        if self is not None:
            nodes.append(self)
        if self.left is not None:
            nodes.extend(self.left.get_nodes())
        if self.right is not None:
            nodes.extend(self.right.get_nodes())
        return nodes

def get_tensor_tree(node):
    _get_tensor_tree(node, node.indices)
    return node.tensor

def _get_tensor_tree(node, indices):
    ### generate the tensor tree for the two sub trees
    if node.left:
        _get_tensor_tree(node.left, indices)
    if node.right:
        _get_tensor_tree(node.right, indices)

    ### left and right are now tensors
    # we use the index information during transforming the variables
    if node.get_type() == "index":
        pass

    # variables just get the indices as lists for upper and lower
    elif node.get_type() == "variable":
        node.tensor = TensorTree(f"Var_{node.get_name()}")
        # special case matricies
        if node.left and len(node.left.get_name()) == 2:
            # special case A[i,i]
            node.tensor.upper = [indices[node.left.get_name()[0]]]
            node.tensor.lower = [indices[node.left.get_name()[1]]]
            if node.left.get_name()[0] == node.left.get_name()[1]:
                node.tensor = TensorTree("diag", node.tensor)
        else:
            node.tensor.upper = [indices[index] for index in node.left.get_name()] if node.left else []

    elif node.get_type() == "scalar":
        node.tensor = Scalar(node.get_name())

    elif node.get_type() == "function":
        # CASE 'sqrt'
        if node.get_name() == "sqrt":
            node.tensor = TensorTree('.^', node.left.tensor, Scalar(0.5))
        else:
            node.tensor = TensorTree(node.get_name(), node.left.tensor)

    elif node.get_type() == "sum":
        sum_indices = [indices[index] for index in node.left.get_name()]
        right_upper = node.right.tensor.upper
        for index in sum_indices:
            one = Vector(1)
            one.changeIndex(one.upper[0], index)
            if index in right_upper:
                one = TensorTree("T", one)
            node.tensor = TensorTree("R*", one, node.right.tensor)

    elif node.get_type() == "operation":
        # CASE ^
        if node.get_name() == '^':
            if node.right.tensor.isScalar():
                node.tensor = TensorTree(".^", node.left.tensor, node.right.tensor)
            else:
                raise SemanticException("Right argument of '^' should be a scalar.")
        # CASE unary or pointwise operation
        elif node.get_name() in ['+', '-', '*', '/']:
            # CASE unary operation
            if not node.right:
                node.tensor = TensorTree(node.get_name(), node.left.tensor)
            else:
                # adjust the same indices
                for index in get_same_indices(node.left, node.right):
                    if has_different_position(index, node.left, node.right):
                        # heuristic, use the node with less indices
                        if len(get_indices(node.left)) < len(get_indices(node.right)):
                            adjust_index(index, node.left)
                        else:
                            adjust_index(index, node.right)
                if node.get_name() in ['+', '-']:
                    # adjust the different indices
                    for index in get_different_indices(node.left, node.right):
                        if index in get_indices(node.left):
                            add_index(index, node.right, upper=is_upper(index, node.left))
                        else:
                            add_index(index, node.left, upper=is_upper(index, node.right))
                    assert is_index_equal(node.left, node.right)
                operation = f"R{node.get_name()}" if node.get_name() in ['*', '/'] else f"t{node.get_name()}"
                node.tensor = TensorTree(operation, node.left.tensor, node.right.tensor)
        else:
            raise SemanticException(f"Operation {node.get_name()} not known.")
    else:
        raise SemanticException(f"Operation {node.get_name()} ({node.get_type}) not know.")


def get_indices(node):
    """
    Returns the index set for a node
    :param node: TensorTree
    :return: list
    """
    if isinstance(node, TensorTree):
        return node.scope
    else:
        raise Exception(f"Node type not understand: {type(node)}.")

def get_same_indices(node_a, node_b):
    """
    Calculates the intersection of all indices of A and B.
    :param node_a: TensorTree
    :param node_b: TensorTree
    :return: list
    """
    indices_a = get_indices(node_a)
    indices_b = get_indices(node_b)
    return [indices for indices in indices_a.union(indices_b) if indices in indices_a and indices in indices_b]
    return [indices for indices in set(indices_a + indices_b) if indices in indices_a and indices in indices_b]

def get_different_indices(node_a, node_b):
    """
    Calculates the indices, that are only in A or B and are not common in both.
    :param node_a: TensorTree
    :param node_b: TensorTree
    :return: frozenset
    """
    indices_a = get_indices(node_a)
    indices_b = get_indices(node_b)
    indices = []
    for i in indices_a:
        if i not in indices_b and not i in indices:
            indices.append(i)
    for i in indices_b:
        if i not in indices_a and not i in indices:
            indices.append(i)
    return indices


def is_upper(index, node):
    if isinstance(node, TensorTree):
        return index in node.tensor.upper
    elif isinstance(node, TensorTree):
        return index in node.upper
    else:
        raise Exception(f"Node type not understand: ({type(node)}).")


def is_lower(index, node):
    if isinstance(node, TensorTree):
        return index in node.tensor.lower
    elif isinstance(node, TensorTree):
        return index in node.lower
    else:
        raise Exception(f"Node type not understand: ({type(node)}).")


def has_different_position(index, node_a, node_b):
    if index in node_a.tensor.upper and index in node_b.tensor.lower:
        return True
    elif index in node_a.tensor.lower and index in node_b.tensor.upper:
        return True
    else:
        return False

def is_index_equal(node_a, node_b):
    """
    Calculates if the indice sets of A and B are equal and in the same alignment
    :param node_a: TensorTree or TensorTree
    :param node_b: TensorTree or TensorTree
    :return: boolean
    """
    if isinstance(node_a, TensorTree) and isinstance(node_b, TensorTree):
        return node_a.tensor.upper == node_b.tensor.upper and node_a.tensor.lower == node_b.tensor.lower
    elif isinstance(node_a, TensorTree) and isinstance(node_b, TensorTree):
        return node_a.upper == node_b.upper and node_a.lower == node_b.lower
    else:
        raise Exception(f"Node type not understand: ({type(node_a)}, {type(node_b)}).")

def adjust_index(index, node):
    """
    Switches the index for the tensor of node
    :param index: int
    :param node: TensorTree
    :return: None
    """
    # special case vector
    if is_vector(node) and index in get_indices(node):
        # we can simply change the index
        node.tensor = TensorTree("T", node.tensor)
    else:
        if is_upper(index, node):
            delta = TensorTree("delta", lower=[index, index])
            upper = [i for i in node.tensor.upper if not index in node.tensor.upper]
            lower = node.tensor.lower.copy() if node.tensor.lower else []
            lower.append(index)
            node.tensor = TensorTree("t*", node.tensor, delta, upper=upper, lower=lower)
        else:
            delta = TensorTree("delta", upper=[index, index])
            upper = node.tensor.upper.copy() if node.tensor.upper else []
            upper.append(index)
            lower = [i for i in node.tensor.lower if not index in node.tensor.lower]
            node.tensor = TensorTree("t*", node.tensor, delta, upper=upper, lower=lower)

def add_index(index, node, upper=True):
    one = Vector(1)
    one.changeIndex(0, index)
    if not upper:
        one = TensorTree("T", one)
    node.tensor = TensorTree("R*", one, node.tensor)

def is_scalar(node):
    return len(get_indices(node)) == 0

def is_vector(node):
    return len(get_indices(node)) == 1

def is_matrix(node):
    return len(get_indices(node)) == 2


def generate_tensor_tree(tensor_node, file_name=os.path.join("img","tensor_tree"), view=False):
    def _get_tensor_nodes(tensor_node):
        nodes = []
        if tensor_node:
            nodes.append(tensor_node)
        if tensor_node.left:
            for node in _get_tensor_nodes(tensor_node.left):
                nodes.append(node)
        if tensor_node.right:
            for node in _get_tensor_nodes(tensor_node.right):
                nodes.append(node)
        return nodes
    dot = Digraph(format="pdf")
    nodes_to_number = {}
    i = 0
    for node in _get_tensor_nodes(tensor_node):
        nodes_to_number[node] = str(i)
        name = f"{node.name} ({node.upper}, {node.lower})"
        dot.node(str(i), name)
        i += 1
    for node in _get_tensor_nodes(tensor_node):
        if node.left is not None:
            dot.edge(nodes_to_number[node], nodes_to_number[node.left])
        if node.right is not None:
            dot.edge(nodes_to_number[node], nodes_to_number[node.right])
    dot.render(file_name)
    return dot


operation_to_numpy = {
    "sin": "np.sin",
    "cos": "np.cos",
    "exp": "np.exp",
    "log": "np.log",
    "norm2": "np.linalg.norm",
    "tr": "np.trace",
    "T": "np.transpose",
    "sign": "np.sign",
    "det": "np.linalg.det",
    "inv": "np.linalg.inv",
    "sqrt": "np.sqrt",
    "abs": "np.abs",
    "diag": "np.diag",
    "diag2": "np.diag",
    ".^": "np.power"
}

number_to_alpha = dict(enumerate(string.ascii_lowercase))

def to_einsum_python(vectree):
    index_to_shape = dict()
    variables = vectree.get_variables()
    for alpha, numeric in vectree.indices.items():
        for variable, keys in variables.items():
            # we skip scalar variables
            if not "left" in keys:
                continue
            # all indices are saved in left
            if alpha in keys['left']:
                index_to_shape[vectree.indices[alpha]] = f"{variable}_shape[{keys['left'].index(alpha)}]"
                break
    if not vectree.tensor:
        t = get_tensor_tree(vectree)
    return _to_einsum_python_helper(deepcopy(vectree.tensor), index_to_shape)


def _to_einsum_python_helper(tensor, index_to_shape):
    if tensor.name in ['t+', 't-']:
        # try to check if we can use linear algebra operations and no einsum
        lina_string = _check_linear_algebra_possibility(tensor, index_to_shape)
        if lina_string:
            return lina_string
        else:
            operation = 'np.add' if tensor.name == 't+' else 'np.subtract'
            if not is_index_equal(tensor.left, tensor.right):
                # index sets have the same indices, but may wrong order
                left_indices = "".join([number_to_alpha[i] for i in get_indices(tensor.left)])
                right_indices = "".join([number_to_alpha[i] for i in get_indices(tensor.right)])
                # transform the left side accordingly
                left = f"np.einsum('{left_indices}->{right_indices}', {_to_einsum_python_helper(tensor.left, index_to_shape)})"
            else:
                left = _to_einsum_python_helper(tensor.left, index_to_shape)
            return f'{operation}({left}, {_to_einsum_python_helper(tensor.right, index_to_shape)})'
    elif tensor.name in ['t*', 't/']:
        # try to check if we can use linear algebra operations and no einsum
        lina_string = _check_linear_algebra_possibility(tensor, index_to_shape)
        if lina_string:
            return lina_string
        else:
            # If one of the sides is a var like A[i,j,i] the double indices are already removed
            result_indices = []
            for i in tensor.left.upper + tensor.left.lower:
                # TODO: the if is only for A^i_i, does create an error for delta_ii, but there is no test for the first case, can this occur?
                #if not i in result_indices:
                result_indices.append(i)
            left_indices = "".join([number_to_alpha[i] for i in result_indices])
            result_indices = []
            for i in tensor.right.upper + tensor.right.lower:
                # TODO: the if is only for A^i_i, does create an error for delta_ii, but there is no test for the first case, can this occur?
                #if not i in result_indices:
                result_indices.append(i)
            right_indices = "".join([number_to_alpha[i] for i in result_indices])
            # indices for the resulting tensor
            result_indices = "".join([number_to_alpha[i] for i in get_indices(tensor)])
            # if we divide we calculate 1/right
            if tensor.name == "t/":
                right = f"np.divide(1,{_to_einsum_python_helper(tensor.right, index_to_shape)})"
            else:
                right = _to_einsum_python_helper(tensor.right, index_to_shape)
            return f"np.einsum('{left_indices},{right_indices}->{result_indices}'," \
                   f"{_to_einsum_python_helper(tensor.left,index_to_shape)}, {right})"
    elif tensor.name == "u-":
        return f'np.negative({_to_einsum_python_helper(tensor.left, index_to_shape)})'
    elif tensor.name in operation_to_numpy.keys() or tensor.name == '.^':
        if tensor.right:
            return f'{operation_to_numpy[tensor.name]}' \
                   f'({_to_einsum_python_helper(tensor.left, index_to_shape)}, ' \
                   f'{_to_einsum_python_helper(tensor.right, index_to_shape)})'
        else:
            # CASE transpose of vector is ignored
            if tensor.name == "T":
                if is_vector(tensor.left):
                    return _to_einsum_python_helper(tensor.left, index_to_shape)
            # CASE not a transpose of a vector
            return f'{operation_to_numpy[tensor.name]}({_to_einsum_python_helper(tensor.left, index_to_shape)})'
    elif tensor.name.startswith("Var"):
        indices = get_indices(tensor)
        n_indices = len(indices)
        var_name = tensor.name[4:]
        # check if it is not a scalar, but a number, e.g vector of ones
        if n_indices > 0 and all([i in string.digits for i in var_name]):
            shape = []
            for index in indices:
                shape.append(index_to_shape[index])
            # create a multidimensional object with just ones and multiply the value
            return f"np.ones({','.join(shape)})*{var_name}"
        else:
            # check if there are indices multiple times inside of the variable
            result_indices = []
            for i in tensor.upper:
                if not i in result_indices:
                    result_indices.append(i)
            # 'einsum-diag-trick'
            if len(result_indices) != len(tensor.upper):
                start_indices = "".join([number_to_alpha[i] for i in tensor.upper])
                result_indices = "".join([number_to_alpha[i] for i in result_indices])
                return f"np.einsum('{start_indices}->{result_indices}',{var_name})"
            else:
                return var_name
    elif tensor.name == "delta":
        shape = []
        for index in get_indices(tensor):
            shape.append(index_to_shape[index])
        return f"np.eye({','.join(shape)})"
    else:
        raise Exception("Error during einsum calculation.")

    
def _check_linear_algebra_possibility(tensor, index_to_shape):
    """
    Check if it is possible to use linear algebra operations to perfom the calculation
    :param tensor: TensorTree
    :param index_to_shape: dict
    :return: String
    """
    # we can broadcast skalar operations
    if is_scalar(tensor.left) or is_scalar(tensor.right):
        operation = "np.multiply" if tensor.name == "t*" else "np.divide" \
            if tensor.name == "t/" else "np.add" if tensor.name == "t+" else "np.subtract"
        return f"{operation}({_to_einsum_python_helper(tensor.left, index_to_shape)}, " \
               f"{_to_einsum_python_helper(tensor.right, index_to_shape)})"

    # we can broadcast pointwise operations if indices are equal
    if is_index_equal(tensor, tensor.left) or is_index_equal(tensor, tensor.right):
        # may transposing one sides helps
        if tensor.name in ['t*', 't/', 't-', 't+']:
            operation = "np.multiply" if tensor.name == "t*" else "np.divide" \
                if tensor.name == "t/" else "np.add" if tensor.name == "t+" else "np.subtract"
            # [i,j], [i,j]
            if tensor.left.upper == tensor.right.upper and tensor.left.lower == tensor.right.lower:
                return f'{operation}({_to_einsum_python_helper(tensor.left, index_to_shape)}, ' \
                       f'{_to_einsum_python_helper(tensor.right, index_to_shape)})'
            # [i,j], [j,i]
            elif tensor.left.upper == tensor.right.lower and tensor.left.lower == tensor.right.upper:
                # transpose left
                return f'{operation}(np.transpose({_to_einsum_python_helper(tensor.left, index_to_shape)}), ' \
                       f'{_to_einsum_python_helper(tensor.right, index_to_shape)})'

    # if left and right are of a higher dimension we can not continue
    if len(tensor.left.upper + tensor.left.lower) > 2 or len(tensor.right.upper + tensor.right.lower) > 2:
        return None

    # if result is of a higher dimension we can not continue
    if len(tensor.upper + tensor.lower) > 2:
        return None

    if tensor.name in ['t*', 't/']:
        operation = "np.multiply" if tensor.name == "t*" else "np.divide"
        # CASE one of the nodes is a vector
        if is_vector(tensor.left) or is_vector(tensor.right):
            same_indices = get_same_indices(tensor.left, tensor.right)
            # CASE we have one common index
            if len(same_indices) == 1:
                assert tensor.name == "t*"
                # CASE sum e.g. A_i^j v^i
                if  not same_indices[0] in get_indices(tensor):
                    # CASE both are vectors (sum)
                    if is_vector(tensor.left) and is_vector(tensor.right):
                        return f"np.sum({operation}({_to_einsum_python_helper(tensor.left, index_to_shape)}," \
                               f"{_to_einsum_python_helper(tensor.right, index_to_shape)}))"
                    # CASE one is a vector, one is a matrix (scalars are already done above) (sum)
                    else:
                        left = _to_einsum_python_helper(tensor.left, index_to_shape)
                        right = _to_einsum_python_helper(tensor.right, index_to_shape)
                        if is_matrix(tensor.left):
                            if is_upper(same_indices[0], tensor.left):
                                tmp = right
                                right = left
                                left = tmp
                        else:
                            if is_lower(same_indices[0], tensor.right):
                                tmp = right
                                right = left
                                left = tmp
                        return f"np.dot({left}, {right})"


                else:
                    left = _to_einsum_python_helper(tensor.left, index_to_shape)
                    right = _to_einsum_python_helper(tensor.right, index_to_shape)
                    # CASE left is a vector
                    if is_vector(tensor.left):
                        if is_upper(same_indices[0], tensor.right):
                            return f"np.transpose({operation}({left},np.transpose({right})))"
                        else:
                            return f"{operation}({left},{right})"
                    # CASE right is a vector
                    else:
                        if is_upper(same_indices[0], tensor.left):
                            return f"np.transpose({operation}(np.transpose({left}),{right}))"
                        else:
                            return f"{operation}({left},{right})"

            # CASE both are vectors
            elif is_vector(tensor.left) and is_vector(tensor.right):
                left = _to_einsum_python_helper(tensor.left, index_to_shape)
                right = _to_einsum_python_helper(tensor.right, index_to_shape)
                if tensor.name in ['t*', 't/']:
                    if tensor.name == "t/":
                        right = f"np.divide(1.0,{right})"
                    # we can use an outer product
                    upper = tensor.upper[0]
                    if upper in get_indices(tensor.left):
                        return f"np.outer({left}, {right})"
                    else:
                        return f"np.outer({right}, {left})"
        # CASE both are matricies are not possible. On one side they pointwise operations are already performed
        # on the other side it is not possible to get a dot product (we do this via vectors).
        else:
            pass
    return None


class Scalar(TensorTree):
    def __init__(self, name):
        TensorTree.__init__(self, 'Var_' + str(name))