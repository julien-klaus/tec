
import argparse

from tec.parser import Parser
from tec.backend import generate


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Translate textbook formulas into Einsum.")
    parser.add_argument("expression", type=str, help="textbook formula for the translation, e.g. sum[i](x[i]*A[i,j])")
    parser.add_argument("-l", "--library", default="numpy", type=str, choices=["numpy", "tensorflow", "pytorch"], help="translate into numpy (default), pytorch or tensorflow")
    parser.add_argument("-v", "--verbose", action="store_true", help="shows the intermediate program code for verifying the translation")
    args = parser.parse_args()

    expression = args.expression
    library = args.library.lower()
    
    print(generate(expression, library, verbose=args.verbose))

