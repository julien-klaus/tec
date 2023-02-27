from tec.parser import Parser
from tec.verification import Verification


function_to_numpy = {
    "sin": "np.sin",
    "cos": "np.cos",
    "exp": "np.exp",
    "log": "np.log",
    "sign": "np.sign",
    "sqrt": "np.sqrt",
    "abs": "np.abs",
    ".^": "np.power",
    "sum": "np.einsum"
}


def to_einsum(tt):
    # recursivly create the python code
    if tt.get_type() == "index":
        return ""
    elif tt.get_type() == "variable":
        return tt.get_name()
    elif tt.get_type() == "scalar": 
        return tt.get_name()
    elif tt.get_type() == "function":
        # binary operations function(left)#
        return f"{function_to_numpy[tt.get_name()]}({to_einsum(tt.left)})"
    elif tt.get_type() == "sum":
        einsum_string = "".join(tt.right.index) + "->" + "".join(tt.index)
        return f"{function_to_numpy[tt.get_name()]}('{einsum_string}',{to_einsum(tt.right)})"
    elif tt.get_type() == "operation":
        # special case unary operations
        if tt.right is None:
            return tt.get_name() + to_einsum(tt.left)
        # special case ^
        elif tt.get_name() == "^":
            return to_einsum(tt.left) + "**" + to_einsum(tt.right)
        # special case, we do this using einsum
        elif tt.get_name() in ["*", "/"]:
            einsum_string = "".join(tt.left.index) + "," + "".join(tt.right.index) + "->" + "".join(tt.index)
            # compute 1 / x
            right_operand = to_einsum(tt.right)
            if tt.get_name() == "/":
                right_operand = "1/" + right_operand
            return f"np.einsum('{einsum_string}',{to_einsum(tt.left)},{right_operand})"
        else:
            # adjust the dimenions
            left_operand = to_einsum(tt.left)
            missing_left = [i for i in tt.right.index if not i in tt.left.index]
            right_operand = to_einsum(tt.right)
            missing_right = [i for i in tt.left.index if not i in tt.right.index]
            # determine the einsum output string
            output_indices = "".join(tt.index)
            if len(missing_left) > 0 and len(missing_right) == 0:
                output_indices = "".join(tt.right.index)
            if len(missing_right) > 0 and len(missing_left) == 0:
                output_indices = "".join(tt.left.index)
            # first left
            if len(missing_left) > 0 and tt.left.get_type() != "scalar":
                einsum_string = "".join(tt.left.index) + "," + "".join(missing_left) + "->" + output_indices
                one_tensor = f"np.ones(({','.join([tt.index_to_shape[index] for index in missing_left])}))"
                left_operand = f"np.einsum('{einsum_string}',{left_operand},{one_tensor})"
            # then right
            if len(missing_right) > 0 and tt.right.get_type() != "scalar":
                einsum_string = "".join(tt.right.index) + "," + "".join(missing_right) + "->" + output_indices
                one_tensor = f"np.ones(({','.join([tt.index_to_shape[index] for index in missing_right])}))"
                right_operand = f"np.einsum('{einsum_string}',{right_operand},{one_tensor})"
            return left_operand + tt.get_name() + right_operand
    else:
        raise Exception(f"Could not create python code. Node type '{tt.get_type()}'")



class NumpyGenerator():
    def __init__(self, input_string):
        self.input_string = input_string
        self.parser_obj = Parser(input_string)

    def get_shapes(self):
        equal_shapes_index = dict()
        variables = self.parser_obj.tensortree.get_variables()
        for alpha, numeric in self.parser_obj.tensortree.indices.items():
            equal_shapes_index[alpha] = []
            for variable, keys in variables.items():
                # all indices are saved in left
                if 'left' in keys and alpha in keys['left']:
                    equal_shapes_index[alpha].append(f"{variable}.shape[{keys['left'].index(alpha)}]")
        return equal_shapes_index

    def generate(self, indent='   ', verbose=False):
        shapes = self.get_shapes()
        code = to_einsum(self.parser_obj.tensortree)

        v = Verification(self.parser_obj.tensortree, code)
        if not v.validate(verbose=verbose):
            raise Exception("Error during verification of generated code. "
                            "We can not compile this expression.")

        
        variables = self.parser_obj.tensortree.get_variables()
        s = f'''def rename_this_function({", ".join(variables.keys())}):\n'''
        s += f'{indent}"""\n'
        s += f'{indent}Generated with TEC from input:\n{indent}{indent}{self.input_string}\n'
        s += f'{indent}Matching matrix and vector dimensions:\n'
        for index, shapes in shapes.items():
            if len(shapes) > 1:
                s+= f'{indent}{indent}{" == ".join(shapes)}\n'
        s += f'{indent}"""\n'
        s+= f'{indent}return {code}'
        return s
