from tec.parser import Parser
from tec.verification import Verification


function_to_numpy = {
    "sin": "np.sin",
    "cos": "np.cos",
    "exp": "np.exp",
    "log": "np.log",
    "tanh": "np.tanh",
    "arcsin": "np.arcsin",
    "arccos": "np.arccos",
    "arctan": "np.arctan",
    "sign": "np.sign",
    "sqrt": "np.sqrt",
    "abs": "np.abs",
    ".^": "np.power",
    "sum": "np.einsum",
    "ones": "np.ones",
    "identity": ""
}

function_to_torch = {
    "sin": "pt.sin",
    "cos": "pt.cos",
    "exp": "pt.exp",
    "log": "pt.log",
    "tanh": "pt.tanh",
    "arcsin": "pt.arcsin",
    "arccos": "pt.arccos",
    "arctan": "pt.arctan",
    "sign": "pt.sign",
    "sqrt": "pt.sqrt",
    "abs": "pt.abs",
    ".^": "pt.pow",
    "sum": "pt.einsum",
    "ones": "pt.ones",
    "identity": "pt.tensor"
}

function_to_tensorflow = {
    "sin": "tf.math.sin",
    "cos": "tf.math.cos",
    "exp": "tf.math.exp",
    "log": "tf.math.log",
    "tanh": "tf.tanh",
    "arcsin": "tf.math.asin",
    "arccos": "tf.math.acos",
    "arctan": "tf.math.atan",
    "sign": "tf.math.sign",
    "sqrt": "tf.math.sqrt",
    "abs": "tf.math.abs",
    ".^": "tf.math.pow",
    "sum": "tf.einsum",
    "ones": "tf.ones",
    "identity": "tf.constant"
}

def replace_index_einsum(expression, new_indices):
    i = 0
    code = ""
    while i < len(expression):
        # lets look for the ->
        if expression[i] == "-":
            code += expression[i]
            i += 1
            if expression[i] == ">":
                code += expression[i]
                i += 1
                # here comes the new index string
                code += new_indices
            else:
                raise Exception("This is not an einsum string. May there is an error. Please report your expression.")
            while expression[i] != "'":
                i += 1
            code += expression[i:]
            break
        else:
            code += expression[i]
            i += 1
    return code

def to_einsum(tt, function_dict):
    # recursivly create the python code
    if tt.get_type() == "index":
        return ""
    elif tt.get_type() == "variable":
        return tt.get_name()
    elif tt.get_type() == "scalar": 
        if function_dict["identity"] != "":
            # torch needs the data type, since it can not cast it correctly
            if function_dict["identity"] == "tf.constant":
                return f'{function_dict["identity"]}({tt.get_name()}, dtype="double")'
            else:
                return f'{function_dict["identity"]}({tt.get_name()})'
        else:
            return tt.get_name()
    elif tt.get_type() == "function":
        # binary operations function(left)#
        return f"{function_dict[tt.get_name()]}({to_einsum(tt.left, function_dict)})"
    ##########################
    # special case summation #
    ##########################
    elif tt.get_type() == "sum":
        ################################
        # merge sum and multiplication #
        ################################
        if tt.right.get_name() in ["*", "/"]:
            right_code = to_einsum(tt.right, function_dict)
            # this should return an einsum operation
            assert right_code[3:].startswith("einsum"), "Could not compile this expression. Please report your expression."
            # since we sum we only have to change the left side of the arrow
            right_code = replace_index_einsum(right_code, "".join(tt.index))
            return right_code
        else:
            einsum_string = "".join(tt.right.index) + "->" + "".join(tt.index)
            return f"{function_dict[tt.get_name()]}('{einsum_string}',{to_einsum(tt.right, function_dict)})"
    elif tt.get_type() == "operation":
        # special case unary operations
        if tt.right is None:
            return tt.get_name() + to_einsum(tt.left, function_dict)
        # special case ^
        elif tt.get_name() == "^":
            return "(" + to_einsum(tt.left, function_dict) + ")" + "**" + to_einsum(tt.right, function_dict)
        #########################################
        # special case, we do this using einsum #
        #########################################
        elif tt.get_name() in ["*", "/"]:
            einsum_string = "," + "".join(tt.right.index) 
            result_index = "".join(tt.index)
            operands = []
            # compute 1 / x
            right_operand = to_einsum(tt.right, function_dict)
            if tt.get_name() == "/":
                right_operand = "1/" + right_operand
            operands.append(right_operand)
            ####################################################
            # merge all multiplications into one einsum string #
            ####################################################
            while tt.left.get_name() in ["*", "/"]:
                tt = tt.left
                einsum_string += "," + "".join(tt.right.index)
                # compute 1 / x
                right_operand = to_einsum(tt.right, function_dict)
                if tt.get_name() == "/":
                    right_operand = "1/" + right_operand
                operands.append(right_operand)
            einsum_string = "".join(tt.left.index) + einsum_string + "->" + result_index
            return f"{function_dict['sum']}('{einsum_string}',{to_einsum(tt.left, function_dict)},{','.join(operands)})"
        else:
            # adjust the dimenions
            left_operand = to_einsum(tt.left, function_dict)
            missing_left = [i for i in tt.right.index if not i in tt.left.index]
            right_operand = to_einsum(tt.right, function_dict)
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
                one_tensor = f"{function_dict['ones']}(({','.join([tt.index_to_shape[index] for index in missing_left])}))"
                left_operand = f"{function_dict['sum']}('{einsum_string}',{left_operand},{one_tensor})"
            # then right
            if len(missing_right) > 0 and tt.right.get_type() != "scalar":
                einsum_string = "".join(tt.right.index) + "," + "".join(missing_right) + "->" + output_indices
                one_tensor = f"{function_dict['ones']}(({','.join([tt.index_to_shape[index] for index in missing_right])}))"
                right_operand = f"{function_dict['sum']}('{einsum_string}',{right_operand},{one_tensor})"
            return left_operand + tt.get_name() + right_operand
    else:
        raise Exception(f"Could not create python code. Node type '{tt.get_type()}'")




def get_shapes(tensortree):
    equal_shapes_index = dict()
    variables = tensortree.get_variables()
    for alpha, numeric in tensortree.indices.items():
        equal_shapes_index[alpha] = []
        for variable, keys in variables.items():
            # all indices are saved in left
            if 'left' in keys and alpha in keys['left']:
                equal_shapes_index[alpha].append(f"{variable}.shape[{keys['left'].index(alpha)}]")
    return equal_shapes_index


def generate_backend(backend, tensortree, indent='   ', verbose=False):
    """Generate Backend Code with Variables and Import"""
    
    # generate import
    _code = ""
    if backend in ["numpy", "np"]:
        function_dict = function_to_numpy
        _code += "#########\n"
        _code += "# NumPy #\n"
        _code += "#########\n"
        _code += "import numpy as np\n\n"
        variable_initialize_function = "np.random.rand"
        result_variable = "result_numpy"
    elif backend in ["pytorch", "torch", "pt"]:
        function_dict = function_to_torch
        _code += "###########\n"
        _code += "# PyTorch #\n"
        _code += "###########\n"
        _code += "import torch as pt\n\n"
        variable_initialize_function = "pt.rand"
        result_variable = "result_pytorch"
    elif backend in ["tensorflow", "tf"]:
        function_dict = function_to_tensorflow
        _code += "##############\n"
        _code += "# TensorFlow #\n"
        _code += "##############\n"
        _code += "import tensorflow as tf\n\n"
        variable_initialize_function = "tf.random.uniform"
        result_variable = "result_tensorflow"
    else:
        raise NotImplementedError("Backend not supported.")
    
    # initialize indices
    indices = get_shapes(tensortree)
    _code += ",".join(indices.keys()) + " = " + ",".join(['3']*len(indices)) + "\n\n"

    # initialize variables
    variables = tensortree.get_variables()
    for variable, indices in variables.items():
        if backend in ["tensorflow", "tf"]:
            _code += f"{variable} = {variable_initialize_function}([{','.join(indices['left'])}])\n"
        else:
            _code += f"{variable} = {variable_initialize_function}({','.join(indices['left'])})\n"
    _code += "\n"
        
    # generate code
    if backend in ["tensorflow", "tf"]:
        _code += "# tensorflow does not cast automatically, probably you have to adjust the dtype of tf.constant\n"
    code = to_einsum(tensortree, function_dict)
    _code += f"{result_variable} = {code}"

    if verbose:
        print("Generated Einsum Code:", code, end="\n\n")
    
    # verify the generated code
    v = Verification(tensortree, to_einsum(tensortree, function_to_numpy))
    if not v.validate(verbose=verbose):
        raise Exception("Error during verification of generated code. "
                        "We can not compile this expression.")

    return _code

    

def generate(expression, backend, indent='   ', verbose=False):
    """Generate Python code for an expression using NumPy, PyTorch or Tensorflow.

    Args:
        expression (str): An expression in textbook like format, e.g. 'sum[i](x[i]*y[i])'.
        backend (str): Backend for the code generation. Either Numpy, PyTorch or Tensorflow.
        indent (str, optional): Indents of the code. Defaults to '   '.
        verbose (bool, optional): Show the code during verification. Defaults to False.

    Returns:
        str: Python code.
    """

    parser_obj = Parser(expression)
    backend = backend.lower()
    assert backend in ["numpy", "pytorch", "tensorflow", "np", "torch", "tf", "pt", "all"]

    code = f"# Generated with TEC from input: {expression}\n\n"

    if backend == "all":
        for _backend in ["numpy", "pytorch", "tensorflow"]:
            code += generate_backend(_backend, parser_obj.tensortree, verbose=verbose)
            code += "\n\n"
    else:
        # create computation
        code += generate_backend(backend, parser_obj.tensortree, verbose=verbose)
    
    return code
