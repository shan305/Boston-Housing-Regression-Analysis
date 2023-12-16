#!/usr/bin/env python
# coding: utf-8

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.datasets import load_iris
from joblib import dump, load
from sklearn.utils.validation import check_is_fitted, NotFittedError


def load_boston_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = ['median_income', 'housing_median_age', 'population', 'latitude', 'longitude',
                     'total_rooms', 'total_bedrooms', 'households', 'median_house_value', 'etc.',
                     'etc.', 'etc.', 'etc.']
    california_df = pd.DataFrame(data, columns=feature_names)
    california_df['target'] = target

    return california_df


def explore_and_visualize_data(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()


def prepare_data_for_regression(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree_regressor(X_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2


def visualize_predictions(y_test, predictions, model_name):
    plt.scatter(y_test, predictions)
    plt.title(f'{model_name} - Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()


def fine_tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_rf_model = RandomForestRegressor(random_state=42, **best_params)
    best_rf_model.fit(X_train, y_train)

    return best_rf_model, best_params


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f'The model has been saved to {filename}')


def load_and_predict_new_data(model_filename, X_new, y_new):
    loaded_model = load(model_filename)

    try:
        check_is_fitted(loaded_model)
    except NotFittedError:
        X_train = X_new
        y_train = y_new
        loaded_model.fit(X_train, y_train)

    new_predictions = loaded_model.predict(X_new)
    new_mse = mean_squared_error(y_new, new_predictions)
    new_r2 = r2_score(y_new, new_predictions)

    return new_predictions, new_mse, new_r2


def visualize_new_predictions(y_new, new_predictions):
    plt.scatter(y_new, new_predictions)
    plt.title('New Dataset Predictions')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()


def feature_importance_plot(model, X_columns):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', legend=False)
    plt.title('Feature Importance - Random Forest Regressor')
    plt.show()


# Step 1: Load the Boston Housing Dataset
california_df = load_boston_data()

# Step 2: Explore and Visualize the Data
explore_and_visualize_data(california_df)

# Step 3: Prepare the Data for Regression
X_train, X_test, y_train, y_test = prepare_data_for_regression(california_df)

# Step 4: Train a Linear Regression Model
linear_model = train_linear_regression(X_train, y_train)

# Step 5: Train Decision Tree Regressor Model
dt_model = train_decision_tree_regressor(X_train, y_train)

# Step 6: Train Random Forest Regressor Model
rf_model = train_random_forest_regressor(X_train, y_train)

# Step 7: Make Predictions and Evaluate the Model
linear_mse, linear_r2 = evaluate_model(linear_model, X_test, y_test)
dt_mse, dt_r2 = evaluate_model(dt_model, X_test, y_test)
rf_mse, rf_r2 = evaluate_model(rf_model, X_test, y_test)

print("Linear Regression:")
print(f'Mean Squared Error: {linear_mse}')
print(f'R-squared: {linear_r2}')
print("\nDecision Tree Regressor:")
print(f'Mean Squared Error: {dt_mse}')
print(f'R-squared: {dt_r2}')
print("\nRandom Forest Regressor:")
print(f'Mean Squared Error: {rf_mse}')
print(f'R-squared: {rf_r2}')

# Step 8: Visualize the Predictions for Comparison
visualize_predictions(y_test, linear_model.predict(X_test), 'Linear Regression')
visualize_predictions(y_test, dt_model.predict(X_test), 'Decision Tree Regressor')
visualize_predictions(y_test, rf_model.predict(X_test), 'Random Forest Regressor')

# Step 9: Fine-Tune Hyperparameters for Random Forest Regressor
best_rf_model, best_params = fine_tune_random_forest(X_train, y_train)

# Step 10: Save the Trained Model for Future Use
save_model(best_rf_model, 'best_random_forest_model.joblib')

# Step 11: Feature Importance Plot for Random Forest Regressor
feature_importance_plot(best_rf_model, X_train.columns)

# Step 12: Load and Predict on New Dataset
new_predictions, new_mse, new_r2 = load_and_predict_new_data('best_random_forest_model.joblib', X_test, y_test)

print("\nNew Dataset Predictions:")
print(f'Mean Squared Error: {new_mse}')
print(f'R-squared: {new_r2}')

# Visualize the Predictions on the New Dataset
visualize_new_predictions(y_test, new_predictions)
