
from tec.parser import Parser
from tec.backend import generate


if __name__ == "__main__":

    expression = "sqrt(sum[i,j,k]((A[i,j,k]-sum[n,m,l](Z[n,m,l]*B[n,i]*C[m,j]*D[k,l]))^2))"
    
    print("Generated Numpy Code:")
    print(generate(expression, "numpy", verbose=False))

