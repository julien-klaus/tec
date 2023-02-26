# -*- coding: utf-8 -*-

from tec.tensortree import TenosorTree, to_einsum_python


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
    "logdet": "",  # special case
    "inv": "np.linalg.inv",
    "sqrt": "np.sqrt",
    "abs": "np.abs",
    "diag": "np.diag",
    "+": "np.add",
    "-": "np.subtract",
    "*": "np.multiply",
    "/": "np.divide",
    "^": "np.power"
}


class Verification(object):
    def __init__(self, vectree, tensortree, dim=4):
        self.v = vectree
        self.v_adjusted = None
        self.t = tensortree
        self.dim = dim
        self.variables = vectree.get_variables()
        self.unbound = []

    def validate(self, delta=10e-8, verbose=False):
        self.unbound = self.v.get_unbound_indices()
        self.v_adjusted = self.v.copy()
        for i in self.unbound:
            # save the indices list
            indices = self.v.indices
            self.v_adjusted = VecTree("sum", left=VecTree([i], type="index"), right=self.v_adjusted, type="sum")
            self.v_adjusted.indices = indices
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
        code += self._generate_code_unvec(self.v_adjusted)
        return code

    def _generate_code(self, delta=10e-8):
        # VARIABLES
        code = self._generate_variables()
        # UNVEC
        code += self._generate_code_unvec(self.v_adjusted)
        # VEV
        vec = to_einsum_python(self.v)
        # sum over all resulting indices, check you only some over different indices A[i,i] just needs one sum
        already_summed_indices = []
        for i in self.t.upper:
            if i not in already_summed_indices:
                vec = f"sum({vec})"
                already_summed_indices.append(i)
        for i in self.t.lower:
            if i not in already_summed_indices:
                vec = f"sum({vec})"
                already_summed_indices.append(i)
        # assign the vectoriced code
        code += f"vec = {vec}\n"
        # DELTA
        code += f"valid = np.linalg.norm(np.subtract(unvec,vec)) < {delta}\n"
        code += f"valid = valid or np.linalg.norm(np.subtract(unvec/{self.dim},vec)) < {delta}\n"
        code += f"valid = valid or np.linalg.norm(np.subtract(unvec,vec/{self.dim})) < {delta}\n"
        code += "assert(valid == True)\n"
        # print(code)
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

    def _generate_code_unvec(self, v):
        nodesCount = v.get_number_of_nodes()
        code = "sums = np.zeros(%i)\n" % (nodesCount)  # is obvious to big
        code += "n = %i\n" % (self.dim)
        code += self._generate_code_unvec_helper(v, "", 0)
        code += "unvec = sums[0]\n"
        return code

    def _generate_code_unvec_helper(self, v, level, sumsInd):
        code = ""
        if v is not None:
            # SUM
            if v.get_name() == "sum":
                index = v.left.get_name()[0]
                code += level + "sums[%i] = 0\n" % (sumsInd + 1)
                code += level + "for %s in range(n):\n" % (index)
                tempcode = self._generate_code_unvec_helper(v.right, level + "\t", sumsInd + 2)
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
                    code += self._generate_code_unvec_helper(v.left, level, sumsInd + 1)
                    code += level + "sums[%i] = np.log(np.linalg.det(sums[%i]))\n" % (sumsInd, sumsInd + 1)
                else:
                    code += self._generate_code_unvec_helper(v.left, level, sumsInd + 1)
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
                    code += self._generate_code_unvec_helper(v.left, level, sumsInd + 1)
                    code += level + "sums[%i] = np.negative(sums[%i])\n" % (sumsInd, sumsInd + 1)
                    code += level + "sums[%i] = 0\n" % (sumsInd + 1)
                # BINOP
                else:
                    code += self._generate_code_unvec_helper(v.left, level, sumsInd + 1)
                    code += self._generate_code_unvec_helper(v.right, level, sumsInd + 2)
                    code += level + "sums[%i] = %s(sums[%i],sums[%i])\n" % (
                        sumsInd, operation_to_numpy[v.get_name()], sumsInd + 1, sumsInd + 2)
                    code += level + "sums[%i] = 0\n" % (sumsInd + 1)
                    code += level + "sums[%i] = 0\n" % (sumsInd + 2)

            # skip forall
            elif v.get_name() == "forall":
                code += self._generate_code_unvec_helper(v.right, level, sumsInd)
        return code

    def _get_variable(self, v):
        if v.get_type() == "scalar" or not v.left:
            return v.get_name()
        else:
            # if not a scalar all indices are stored left
            return f"{v.get_name()}[{','.join(v.left.get_name())}]"
