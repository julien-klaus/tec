# -*- coding: utf-8 -*-

from tec.tensortree import TensorTree

operation_to_numpy = {
    "sin": "np.sin",
    "cos": "np.cos",
    "exp": "np.exp",
    "log": "np.log",
    "sign": "np.sign",
    "inv": "np.linalg.inv",
    "sqrt": "np.sqrt",
    "abs": "np.abs",
    "+": "np.add",
    "-": "np.subtract",
    "*": "np.multiply",
    "/": "np.divide",
    "^": "np.power"
}


class Verification(object):
    def __init__(self, tensortree, einsum_code, dim=4):
        self.t = tensortree.copy()
        self.t.flat_sum()
        self.t_adj = None
        self.dim = dim
        self.variables = tensortree.get_variables()
        self.einsum_code = einsum_code
        self.unbound = []

    def validate(self, delta=10e-8, verbose=False):
        self.unbound = self.t.index
        self.t_adjusted = self.t.copy()
        for i in self.unbound:
            # save the indices list
            indices = self.t.indices
            self.t_adjusted = TensorTree("sum", left=TensorTree([i], type="index"), right=self.t_adjusted, type="sum")
            self.t_adjusted.indices = indices
        code = self._generate_code(delta)
        if verbose:
            print(code)
        try:
            exec(code)
        except Exception:
            return False
        return True

    def generate_for_code(self):
        code = self._generate_variables()
        code += self._generate_code_for(self.t_adjusted)
        return code

    def _generate_code(self, delta=10e-8):
        # VARIABLES
        code = self._generate_variables()
        # FOR Code
        code += self._generate_code_for(self.t_adjusted)
        # sum over all resulting indices, check you only some over different indices A[i,i] just needs one sum
        already_summed_indices = []
        for i in self.t.index:
            if i not in already_summed_indices:
                self.einsum_code = f"sum({self.einsum_code})"
                already_summed_indices.append(i)
        # assign the vectoriced code
        code += f"einsum = {self.einsum_code}\n"
        # DELTA
        code += f"valid = np.allclose(einsum, unvec)\n"
        code += f"valid = valid or np.allclose(einsum/n, unvec)\n"
        code += f"valid = valid or np.allclose(einsum, unvec/n)\n"
        code += f"assert valid"
        return code

    def _generate_variables(self):
        code = "import numpy as np\n"
        for var in self.variables:
            if 'left' in self.variables[var]:
                index = self.variables[var]['left']
                code += f"{var} = np.random.rand({','.join([str(self.dim)]*len(index))})\n"
                code += f"{var}_shape = np.shape({var})\n"
            else:
                code += f"{var} = np.random.rand()\n"
        return code

    def _generate_code_for(self, v):
        nodesCount = v.get_number_of_nodes()
        code = "sums = np.zeros(%i)\n" % (nodesCount)  # is obvious to big
        code += "n = %i\n" % (self.dim)
        code += self._generate_code_for_helper(v, "", 0)
        code += "unvec = sums[0]\n"
        return code

    def _generate_code_for_helper(self, v, level, sumsInd):
        code = ""
        if v is not None:
            # SUM
            if v.get_name() == "sum":
                index = v.left.get_name()[0]
                code += level + "sums[%i] = 0\n" % (sumsInd + 1)
                code += level + "for %s in range(n):\n" % (index)
                tempcode = self._generate_code_for_helper(v.right, level + "\t", sumsInd + 2)
                code += tempcode
                # Innersum add
                code += level + "\t" + "sums[%i] = np.add(sums[%i],sums[%i])\n" % (
                sumsInd + 1, sumsInd + 1, sumsInd + 2)
                code += level + "\t" + "sums[%i] = 0\n" % (sumsInd + 2)
                # Outersum add
                code += level + "sums[%i] = sums[%i]\n" % (sumsInd, sumsInd + 1)
                code += level + "sums[%i] = 0\n" % (sumsInd + 1)

            # FUNCTION
            elif v.get_type() == "function":
                if v.get_name() == "logdet":
                    code += self._generate_code_for_helper(v.left, level, sumsInd + 1)
                    code += level + "sums[%i] = np.log(np.linalg.det(sums[%i]))\n" % (sumsInd, sumsInd + 1)
                else:
                    code += self._generate_code_for_helper(v.left, level, sumsInd + 1)
                    code += level + "sums[%i] = %s(sums[%i])\n" % (sumsInd, operation_to_numpy[v.get_name()], sumsInd + 1)

                    # VARIABLE
            elif v.get_type() == "variable":
                var = self._get_variable(v)
                # var can contain +1 or -1, that raises an index out of bound exception
                index = var[var.find("[") + 1:-1].split(",")
                if len(index) > 0 and "" not in index:
                    code += level + "if "
                    for ind in index:
                        code += "%s >= 0 and %s <= %i-1 and " % (ind, ind, self.dim)
                    code += "True:\n"
                    code += level + "\t" + "sums[%i] = %s\n" % (sumsInd, var)
                else:
                    code += level + "sums[%i] = %s\n" % (sumsInd, var)

            # SCALAR
            elif v.get_type() == "scalar":
                code += level + "sums[%i] = np.float64(%s)\n" % (sumsInd, self._get_variable(v))

            # BINOP
            elif v.get_name() in ["+", "-", "*", "/", "^"]:
                # NEGATIVE
                if v.left is not None and v.right is None:
                    code += self._generate_code_for_helper(v.left, level, sumsInd + 1)
                    code += level + "sums[%i] = np.negative(sums[%i])\n" % (sumsInd, sumsInd + 1)
                    code += level + "sums[%i] = 0\n" % (sumsInd + 1)
                # BINOP
                else:
                    code += self._generate_code_for_helper(v.left, level, sumsInd + 1)
                    code += self._generate_code_for_helper(v.right, level, sumsInd + 2)
                    code += level + "sums[%i] = %s(sums[%i],sums[%i])\n" % (
                        sumsInd, operation_to_numpy[v.get_name()], sumsInd + 1, sumsInd + 2)
                    code += level + "sums[%i] = 0\n" % (sumsInd + 1)
                    code += level + "sums[%i] = 0\n" % (sumsInd + 2)

            # skip forall
            elif v.get_name() == "forall":
                code += self._generate_code_for_helper(v.right, level, sumsInd)
        return code

    def _get_variable(self, v):
        if v.get_type() == "scalar" or not v.left:
            return v.get_name()
        else:
            # if not a scalar all indices are stored left
            return f"{v.get_name()}[{','.join(v.left.get_name())}]"
