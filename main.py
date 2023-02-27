
from tec.parser import Parser
from tec.backend import NumpyGenerator


if __name__ == "__main__":
    term = "sum[i,m,l](log(1+exp(-y[i]+sum[j](t[j,l]^2*f[i,j,m]))))"
    ag = NumpyGenerator(term)

    print(ag.generate(verbose=True))

