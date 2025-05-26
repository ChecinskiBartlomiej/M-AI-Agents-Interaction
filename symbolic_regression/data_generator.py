import numpy as np

np.random.seed(42)   
coeff = np.random.uniform(-5,5,10)
X = np.random.normal(0,1,[500,3])

Y = (
    coeff[0]*X[:,0]**2           
  + coeff[1]*X[:,1]**2           
  + coeff[2]*X[:,2]**2           
  + coeff[3]*X[:,0]*X[:,1]       
  + coeff[4]*X[:,0]*X[:,2]       
  + coeff[5]*X[:,1]*X[:,2]       
  + coeff[6]*X[:,0]              
  + coeff[7]*X[:,1]              
  + coeff[8]*X[:,2]              
  + coeff[9]                     
)

feature_names = ['x**2', 'y**2', 'z**2', 'x*y', 'x*z', 'y*z', 'x', 'y', 'z']
terms = [f"{coef:.6f}*{name}" for coef, name in zip(coeff[:-1], feature_names)]
terms.append(f"{coeff[-1]:.6f}")
formula = " + ".join(terms)

data = np.concatenate([X, Y.reshape(-1,1) ], axis=1)

np.savetxt(
    '/home/bc/Desktop/Documents/muchizm/Symbolic_regression/data.csv',
    data,
    delimiter=',',
    fmt='%.6f,%.6f,%.6f,%.6f'  
)


