import numpy as np
from pysr import PySRRegressor
import sympy as sp
import data_generator

data = np.loadtxt('/home/bc/Desktop/Documents/muchizm/Symbolic_regression/data.csv', delimiter=',')
X = data[:, :3] 
Y = data[:, 3]    

    
model = PySRRegressor(
    niterations=100,                    # number of optimization iterations
    binary_operators=["+", "-", "*"],   # allow addition, subtraction, multiplication
    unary_operators=["square"],         # allow squaring to capture x**2, y**2, z**2
    model_selection="best",             # pick the best model by default criteria
    verbosity=1                         # print progress
)

model.fit(X, Y, variable_names=["x", "y", "z"])

best_equation = model.sympy()  
print("Found polynomial:")
print(f"y = {best_equation}")

print("Real polynomial:")
print("y =", data_generator.formula)

# Simplify found polynomial.
x, y, z = sp.symbols('x y z')
expanded = sp.expand(best_equation)
collected = {
    monomial: float(sp.expand(sp.Poly(expanded, x, y, z).coeff_monomial(monomial)))
    for monomial in [x**2, y**2, z**2, x*y, x*z, y*z, x, y, z]
}
constant = float(sp.expand(sp.Poly(expanded, x, y, z).coeff_monomial(1)))
terms = [f"{collected[m]:.6f}*{m}" for m in collected]
terms.append(f"{constant:.6f}")
formula_simplified = " + ".join(terms)
print("Simplified found polynomial:")
print("y =", formula_simplified)




