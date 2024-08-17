import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('bostonHousing.csv')

# Explore the dataset
print(df.head())
print(df.describe())
df.dropna(inplace=True)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1), df['MEDV'], test_size=0.2, random_state=42)

# Implement a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluate the performance of the linear regression model
y_pred = linear_model.predict(X_test)
print('Linear Regression MSE:', mean_squared_error(y_test, y_pred))
print('Linear Regression R²:', r2_score(y_test, y_pred))

# Implement a decision tree regressor model
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

# Evaluate the performance of the decision tree regressor model
y_pred = decision_tree_model.predict(X_test)
print('Decision Tree Regressor MSE:', mean_squared_error(y_test, y_pred))
print('Decision Tree Regressor R²:', r2_score(y_test, y_pred))

# Implement a random forest regressor model
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

# Evaluate the performance of the random forest regressor model
y_pred = random_forest_model.predict(X_test)
print('Random Forest Regressor MSE:', mean_squared_error(y_test, y_pred))
print('Random Forest Regressor R²:', r2_score(y_test, y_pred))

# Compare the performance of the three models using cross-validation
from sklearn.model_selection import cross_val_score
linear_scores = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
decision_tree_scores = cross_val_score(decision_tree_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
random_forest_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

print('Linear Regression Cross-Validation Scores:', linear_scores)
print('Decision Tree Regressor Cross-Validation Scores:', decision_tree_scores)
print('Random Forest Regressor Cross-Validation Scores:', random_forest_scores)

# Select the best model based on the evaluation metrics
best_model = random_forest_model
print('Best Model:', best_model)