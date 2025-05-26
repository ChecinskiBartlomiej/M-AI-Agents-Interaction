import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Noisy-Linear_train.csv',header=None, names=['X', 'Y', 'Z', 'target'])

x = df[['X', 'Y', 'Z']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0],'l1_ratio': [0.1, 0.5, 0.9]} #alpha * (l1_ratio * np.linalg.norm(w, 1) + 0.5 * (1 - l1_ratio) * np.linalg.norm(w, 2)**2)

enet = ElasticNet(max_iter=10000, random_state=42)
grid = GridSearchCV(
    estimator=enet,
    param_grid=param_grid,
    scoring='neg_mean_squared_error', # In scikit learn, bigger = better
    cv=5,
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)

best_enet = grid.best_estimator_

y_train_pred = best_enet.predict(X_train)
y_test_pred  = best_enet.predict(X_test)

print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test  MSE:", mean_squared_error(y_test,  y_test_pred))