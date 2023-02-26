from tec.parser import Parser
from tec.tensortree import to_einsum_python, TensorTree


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

    def generate(self, indent='    '):
        shapes = self.get_shapes()
        code = to_einsum_python(self.parser_obj.tensortree)

        v = Verification(self.parser_obj.tensortree, self.parser_obj.tensortree.tensor)
        if not v.validate():
            raise Exception("Error during verification of generated code. "
                            "We can not calculate this expression.")

        variables = self.parser_obj.tensortree.get_variables()

        s = f'''def rename_this_function({", ".join(variables.keys())}):\n'''
        s += f'{indent}"""\n'
        s += f'{indent}Generated with LinA from input:\n{indent}{indent}{self.input_string}\n'
        s += f'{indent}Matching matrix and vector dimensions:\n'
        for index, shapes in shapes.items():
            if len(shapes) > 1:
                s+= f'{indent}{indent}{" == ".join(shapes)}\n'
        s += f'{indent}"""\n'
        s+= f'return {code}'
        return s
