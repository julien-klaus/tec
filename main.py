
from tec.parser import Parser
from tec.np_backend import NumpyGenerator


if __name__ == "__main__":
    term = "sum[i](log(1+exp(-y[i]+sum[j](t[j]*f[i,j]))))"
    ag = NumpyGenerator(term)
    ag.generate()

    