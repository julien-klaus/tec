
from tec.parser import Parser
from tec.backend import NumpyGenerator


if __name__ == "__main__":

    term = "sqrt(sum[i,j,k]((A[i,j,k]-sum[n,m,l](Z[n,m,l]*B[n,i]*C[m,j]*D[k,l]))^2))"
    
    ag = NumpyGenerator(term)

    print(ag.generate(verbose=True))

