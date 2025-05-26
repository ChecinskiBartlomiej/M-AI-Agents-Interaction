import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv('Noisy-Linear_train.csv',header=None,names=['X', 'Y', 'Z', 'target'])

x = df[['X', 'Y', 'Z']]
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Build full tree, without pruning.
full_tree_reg = DecisionTreeRegressor(criterion='squared_error')
full_tree_reg.fit(x_train, y_train)

y_train_pred_full = full_tree_reg.predict(x_train)
y_test_pred_full  = full_tree_reg.predict(x_test)

train_mse_full = mean_squared_error(y_train, y_train_pred_full)
test_mse_full  = mean_squared_error(y_test,  y_test_pred_full)

train_r2_full  = r2_score(y_train, y_train_pred_full)
test_r2_full   = r2_score(y_test,  y_test_pred_full)

print("\n** Full tree performance **")
print(f"Train MSE: {train_mse_full:.4f}, R2: {train_r2_full:.4f}")
print(f"Test  MSE: {test_mse_full:.4f}, R2: {test_r2_full:.4f}")

# Generate pruning path.
path = full_tree_reg.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# For each alpha, train a pruned regressor.
cv_mean_mse = []
cv_std_mse  = []

for alpha in ccp_alphas:
    dt_reg = DecisionTreeRegressor(criterion='squared_error', ccp_alpha=alpha)
    neg_mse_scores = cross_val_score(
        dt_reg, x_train, y_train,
        cv=5,
        scoring='neg_mean_squared_error',
    )
    mse_scores = -neg_mse_scores
    cv_mean_mse.append(mse_scores.mean())
    cv_std_mse.append(mse_scores.std())

# Plot alpha vs. average CV MSE.
plt.figure(figsize=(8,5))
plt.errorbar(ccp_alphas, cv_mean_mse, yerr=cv_std_mse) 
plt.xlabel("ccp_alpha")
plt.ylabel("Cross-Validation MSE")
plt.title("CV MSE vs. Alpha for Pruning (Regressor)")
plt.show()

# Select alpha that yields the lowest CV MSE.
idx_best = np.argmin(cv_mean_mse)
best_alpha = ccp_alphas[idx_best]
best_mse = cv_mean_mse[idx_best]
print(f"\nChosen alpha = {best_alpha:.4f}, CV MSE = {best_mse:.4f}")

# Retrain a pruned regressor on the entire training set.
pruned_tree_reg = DecisionTreeRegressor(ccp_alpha=best_alpha)
pruned_tree_reg.fit(x_train, y_train)

# Evaluate on train and test sets.
y_train_pred_pruned = pruned_tree_reg.predict(x_train)
y_test_pred_pruned  = pruned_tree_reg.predict(x_test)

train_mse_pruned = mean_squared_error(y_train, y_train_pred_pruned)
test_mse_pruned  = mean_squared_error(y_test,  y_test_pred_pruned)
train_r2_pruned  = r2_score(y_train, y_train_pred_pruned)
test_r2_pruned   = r2_score(y_test,  y_test_pred_pruned)

print("\n** Pruned tree performance **")
print(f"Train MSE: {train_mse_pruned:.4f}, R2: {train_r2_pruned:.4f}")
print(f"Test MSE: {test_mse_pruned:.4f}, R2: {test_r2_pruned:.4f}")

# Initialize the Random Forest.
rf = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',  
    max_features=2,       
    oob_score=True              
)

rf.fit(x_train, y_train)

y_train_pred_rf = rf.predict(x_train)
y_test_pred_rf  = rf.predict(x_test)

# Evaluate.
train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
test_mse_rf  = mean_squared_error(y_test,  y_test_pred_rf)
train_r2_rf  = r2_score(y_train, y_train_pred_rf)
test_r2_rf   = r2_score(y_test,  y_test_pred_rf)

print("\n** Random forest performance **")
print(f"Train MSE: {train_mse_rf:.4f}, R²: {train_r2_rf:.4f}")
print(f"Test  MSE: {test_mse_rf:.4f}, R²: {test_r2_rf:.4f}")
print(f"OOB R²:   {rf.oob_score_:.4f}")

# Define XGBoost parameters for regression.
xgb_reg = XGBRegressor(
    n_estimators=200,
    objective='reg:squarederror',  
    eval_metric='rmse',             
    max_depth=3,                    
    learning_rate=0.1,              
    subsample=0.8,                  
    colsample_bytree=0.8,           
)

# Train the XGBoost regressor.
xgb_reg.fit(x_train, y_train)

# Predict on train & test sets.
y_train_pred_xgb = xgb_reg.predict(x_train)
y_test_pred_xgb  = xgb_reg.predict(x_test)

# Evaluate.
train_mse_xgb = mean_squared_error(y_train, y_train_pred_xgb)
test_mse_xgb  = mean_squared_error(y_test,  y_test_pred_xgb)
train_r2_xgb  = r2_score(y_train, y_train_pred_xgb)
test_r2_xgb   = r2_score(y_test,  y_test_pred_xgb)

print("\n** XGBoost Regressor Performance **")
print(f"Train MSE: {train_mse_xgb:.4f}, R²: {train_r2_xgb:.4f}")
print(f"Test  MSE: {test_mse_xgb:.4f}, R²: {test_r2_xgb:.4f}")
