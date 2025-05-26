import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('Noisy-Linear_train.csv', header=None, names=['X', 'Y', 'Z', 'target'])

x = df[['X', 'Y', 'Z']]
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression(fit_intercept=False)
model.fit(x_train, y_train)

coeffs = model.coef_
a, b, c = coeffs[0], coeffs[1], coeffs[2]
bias = model.intercept_
print("Coefficients:")
print(f" a = {a:.4f}")
print(f" b = {b:.4f}")
print(f" c = {c:.4f}")
print(f" bias = {bias:.4f}")

predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print(mse)

