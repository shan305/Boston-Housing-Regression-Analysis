#!/usr/bin/env python
# coding: utf-8


# **Step 1: Import Necessary Libraries**

# In[1]:


# Importing required libraries
#!pip install scipy
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





# **Step 2: Load the Boston Housing Dataset**

# In[2]:


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape)

# Create a DataFrame with the correct number of columns
feature_names = ['median_income', 'housing_median_age', 'population', 'latitude', 'longitude',
                 'total_rooms', 'total_bedrooms', 'households', 'median_house_value', 'etc.',
                 'etc.', 'etc.', 'etc.']
california_df = pd.DataFrame(data, columns=feature_names)

# Add the target column to the DataFrame
california_df['target'] = target

# Display basic information about the dataset
print(california_df.info())

# Display summary statistics
print(california_df.describe())




# **Step 3: Explore and Visualize the Data**

# In[3]:


# Visualize the correlation matrix
corr_matrix = california_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()



# **Step 4: Prepare the Data for Regression**

# In[4]:


# Select features and target variable
X = california_df.drop('target', axis=1)
y = california_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# **Step 5: Train a Linear Regression Model**

# In[5]:


# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# **Step 6: Train Decision Tree Regressor Model**

# In[6]:


# Initialize the Decision Tree Regressor model
dt_model = DecisionTreeRegressor()

# Train the model
dt_model.fit(X_train, y_train)


# **Step 7: Train Random Forest Regressor Model**
# 
# 

# In[7]:


# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)



# **Step 8: Make Predictions and Evaluate the Model**

# In[8]:


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Make predictions on the test set for all models
linear_pred = linear_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluate the models
linear_mse = mean_squared_error(y_test, linear_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

linear_r2 = r2_score(y_test, linear_pred)
dt_r2 = r2_score(y_test, dt_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Linear Regression:")
print(f'Mean Squared Error: {linear_mse}')
print(f'R-squared: {linear_r2}')
print("\nDecision Tree Regressor:")
print(f'Mean Squared Error: {dt_mse}')
print(f'R-squared: {dt_r2}')
print("\nRandom Forest Regressor:")
print(f'Mean Squared Error: {rf_mse}')
print(f'R-squared: {rf_r2}')


# It looks like we've successfully trained and evaluated our regression models. The metrics we've printed (`Mean Squared Error` and `R-squared`) provide insights into the performance of each model on the test set.
# 
# Here's a brief interpretation of the metrics:
# 
# - **Linear Regression:**
#   - Mean Squared Error (MSE): 24.29
#   - R-squared (R2): 0.67
# 
# - **Decision Tree Regressor:**
#   - Mean Squared Error (MSE): 10.67
#   - R-squared (R2): 0.85
# 
# - **Random Forest Regressor:**
#   - Mean Squared Error (MSE): 8.33
#   - R-squared (R2): 0.89
# 
# In general:
# - Lower MSE values indicate better performance, as it represents the average squared difference between predicted and actual values.
# - R-squared values close to 1.0 suggest a good fit, indicating the proportion of the variance in the dependent variable that is predictable from the independent variables.
# 
# Based on these metrics, it seems like the Decision Tree and Random Forest models are performing better than the Linear Regression model for your specific dataset. Keep in mind that the choice of the best model may also depend on the specific requirements of your application.

# **Step 9: Visualize the Predictions for Comparison**

# In[9]:


# Visualize actual vs. predicted values for all models
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, linear_pred)
plt.title('Linear Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 3, 2)
plt.scatter(y_test, dt_pred)
plt.title('Decision Tree Regressor')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 3, 3)
plt.scatter(y_test, rf_pred)
plt.title('Random Forest Regressor')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()


# ** **
# - This  project incorporates Decision Tree Regressor and Random Forest Regressor models, providing a comparison of their performance with Linear Regression. we can observe how different models handle the prediction task and visually compare their predictions against the actual values. This comparative analysis can help us choose the most suitable model for the specific regression task.
# 
# 

# **Step 10: Fine-Tune Hyperparameters for Random Forest Regressor**
# - In this step, we perform hyperparameter tuning for the Random Forest Regressor using GridSearchCV. We define a parameter grid with different values for the number of estimators, maximum depth, minimum samples split, and minimum samples leaf. GridSearchCV searches through these combinations to find the best set of hyperparameters that minimize the mean squared error. The best parameters are then used to train the final Random Forest Regressor model, and its performance is evaluated on the test set.
# 
# 

# In[10]:


# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
best_rf_model = RandomForestRegressor(random_state=42, **best_params)
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
best_rf_pred = best_rf_model.predict(X_test)

# Evaluate the model with the best parameters
best_rf_mse = mean_squared_error(y_test, best_rf_pred)
best_rf_r2 = r2_score(y_test, best_rf_pred)

print("Best Random Forest Regressor:")
print(f'Best Parameters: {best_params}')
print(f'Mean Squared Error: {best_rf_mse}')
print(f'R-squared: {best_rf_r2}')


# We've fine-tuned the Random Forest Regressor and found the best parameters for the model. The metrics we provided (`Mean Squared Error` and `R-squared`) indicate the performance of the Random Forest Regressor with the optimized hyperparameters.
# 
# Here's a brief interpretation:
# 
# - **Best Random Forest Regressor:**
#   - **Best Parameters:**
#     - `max_depth`: 10
#     - `min_samples_leaf`: 2
#     - `min_samples_split`: 2
#     - `n_estimators`: 100
#   - **Mean Squared Error (MSE):** 9.11
#   - **R-squared (R2):** 0.88
# 
# These results suggest that with the tuned hyperparameters, the Random Forest Regressor is performing well on our dataset, and the model has improved compared to the default settings.
# 

# **Step 11: Feature Importance Plot for Random Forest Regressor**
# - This step involves extracting feature importances from the best Random Forest Regressor model and visualizing them in a bar plot. Understanding feature importances helps identify which features have the most impact on the model's predictions.
# 
# 

# In[11]:


# Get feature importances from the best Random Forest Regressor model
feature_importances = best_rf_model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', legend=False)
plt.title('Feature Importance - Random Forest Regressor')
plt.show()


# This code is designed to visualize the feature importances of the best Random Forest Regressor model. Let me break down the code step by step:
# 
# 1. **`feature_importances = best_rf_model.feature_importances_`**: Extracts the feature importances from the best Random Forest Regressor model (`best_rf_model`). Feature importances represent the contribution of each feature to the model's predictions.
# 
# 2. **`feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})`**: Creates a DataFrame (`feature_importance_df`) with two columns: 'Feature' containing the names of the features (taken from `X.columns`), and 'Importance' containing the corresponding feature importances.
# 
# 3. **`feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)`**: Sorts the DataFrame by the 'Importance' column in descending order. This step is done to arrange the features from the most to the least important.
# 
# 4. **Plotting the Feature Importance**:
#     - **`plt.figure(figsize=(10, 6))`**: Sets the figure size for the plot.
#     - **`sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', legend=False)`**: Creates a bar plot using Seaborn (`sns.barplot`). The x-axis represents the importance of each feature, the y-axis represents the feature names, and each feature is color-coded. The `legend=False` parameter is used to suppress the legend.
#     
#     - **`plt.title('Feature Importance - Random Forest Regressor')`**: Sets the title of the plot.
# 
#     - **`plt.show()`**: Displays the plot.
# 
# The resulting plot visually represents the importance of each feature in the Random Forest Regressor model. Features with higher bars contribute more to the model's predictions. The features are ordered from most important to least important based on their importance values.

# **Step 12: Save the Trained Model for Future Use**

# In[12]:


# Save the best Random Forest Regressor model to a file
model_filename = 'best_random_forest_model.joblib'
joblib.dump(best_rf_model, model_filename)

print(f'The best Random Forest Regressor model has been saved to {model_filename}')


# In[21]:


model_filename = 'best_random_forest_model.joblib'

# Load the trained model
loaded_rf_model = load(model_filename)

# Step 3: Load a New Dataset for Prediction
# For demonstration, let's use the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Select features and target variable
X_new = iris_df.drop('target', axis=1)
y_new = iris_df['target']

# Check if the model is fitted, if not, fit it
try:
    check_is_fitted(loaded_rf_model)
except NotFittedError:
    X_train = iris_df.drop('target', axis=1)
    y_train = iris_df['target']
    loaded_rf_model.fit(X_train, y_train)

# Step 4: Make Predictions on the New Dataset
new_predictions = loaded_rf_model.predict(X_new)

# Step 5: Evaluate the Predictions
new_mse = mean_squared_error(y_new, new_predictions)
new_r2 = r2_score(y_new, new_predictions)

print("New Dataset Predictions:")
print(f'Mean Squared Error: {new_mse}')
print(f'R-squared: {new_r2}')

# Step 6: Visualize the Predictions on the New Dataset
plt.scatter(y_new, new_predictions)
plt.title('Random Forest Regressor - New Dataset Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# Great! It looks like the model has been successfully fitted, and the predictions on the new dataset are providing good results with a low Mean Squared Error and high R-squared value!
# 
# 

# **Extra Step** 
# Lets add this and run our predictions again
# 
# The addition of exploratory data analysis (EDA) and preprocessing steps is a good practice when working on a machine learning project. Here are the reasons behind this addition:
# 
# 1. **Understand the Data:**
#    - EDA helps you gain insights into the characteristics of the dataset.
#    - You can visualize the distribution of features, identify outliers, and understand the relationships between variables.
# 
# 2. **Data Preprocessing:**
#    - It's common to preprocess data before training a machine learning model. This includes handling missing values, scaling features, encoding categorical variables, etc.
#    - Preprocessing ensures that the data is in a suitable format for the machine learning model.
# 
# 3. **Ensure Consistency:**
#    - When sharing code or working collaboratively, it's good to have a consistent structure in your project. Including EDA and preprocessing steps makes the code more comprehensive and understandable for others (or for yourself in the future).
# 
# 4. **Handle Real-world Data Challenges:**
#    - Real-world datasets often come with imperfections. EDA and preprocessing help address these challenges, making the data more robust for modeling.
# 
# 5. **Improve Model Performance:**
#    - Understanding the data and preprocessing it appropriately can lead to better model performance.
#    - For instance, identifying and handling outliers can prevent them from negatively impacting the model's predictions.
# 
# 6. **Best Practices:**
#    - Including EDA and preprocessing aligns with best practices in machine learning development.
#    - It's a good habit to thoroughly understand your data and apply necessary transformations before training a model.
# 
# In summary, the addition of EDA and preprocessing steps enhances the overall quality of your machine learning project. It allows us to build a more robust and reliable model by addressing data challenges and ensuring that the data is well-prepared for training.

# In[22]:


# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# EDA: Display summary statistics and information about the dataset
print("Summary Statistics:")
print(iris_df.describe())

# EDA: Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
plt.hist(iris_df['target'], bins=3, edgecolor='black', alpha=0.7)
plt.title('Distribution of Target Variable')
plt.xlabel('Target Values')
plt.ylabel('Frequency')
plt.show()

# EDA: Visualize relationships between features and target variable
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris['feature_names']):
    plt.subplot(2, 2, i + 1)
    plt.scatter(iris_df[feature], iris_df['target'], edgecolor='black', alpha=0.7)
    plt.title(f'{feature} vs. Target')
    plt.xlabel(feature)
    plt.ylabel('Target')
plt.tight_layout()
plt.show()

# Select features and target variable
X_new = iris_df.drop('target', axis=1)
y_new = iris_df['target']

# Check if the model is fitted, if not, fit it
try:
    check_is_fitted(loaded_rf_model)
except NotFittedError:
    X_train = iris_df.drop('target', axis=1)
    y_train = iris_df['target']
    loaded_rf_model.fit(X_train, y_train)

# Make Predictions on the New Dataset
new_predictions = loaded_rf_model.predict(X_new)

# Evaluate the Predictions
new_mse = mean_squared_error(y_new, new_predictions)
new_r2 = r2_score(y_new, new_predictions)

print("\nNew Dataset Predictions:")
print(f'Mean Squared Error: {new_mse}')
print(f'R-squared: {new_r2}')

# Visualize the Predictions on the New Dataset
plt.figure(figsize=(10, 6))
plt.scatter(y_new, new_predictions)
plt.title('Random Forest Regressor - New Dataset Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# The summary statistics provide an overview of the dataset. Now, let's interpret the key statistics:
# 
# 1. **Count:** There are 150 entries in the dataset, indicating that there are no missing values for any feature.
# 
# 2. **Mean:** The mean values give us the central tendency of the features.
#    - The mean sepal length is approximately 5.84 cm.
#    - The mean sepal width is approximately 3.06 cm.
#    - The mean petal length is approximately 3.76 cm.
#    - The mean petal width is approximately 1.20 cm.
#    - The mean target value is 1.0.
# 
# 3. **Standard Deviation (std):** The standard deviation provides a measure of the dispersion or spread of the values.
#    - The standard deviation of sepal length is approximately 0.83 cm.
#    - The standard deviation of sepal width is approximately 0.44 cm.
#    - The standard deviation of petal length is approximately 1.77 cm.
#    - The standard deviation of petal width is approximately 0.76 cm.
#    - The standard deviation of the target is approximately 0.82.
# 
# 4. **Minimum (min):** The minimum values represent the smallest observed values for each feature.
#    - The minimum sepal length is 4.3 cm.
#    - The minimum sepal width is 2.0 cm.
#    - The minimum petal length is 1.0 cm.
#    - The minimum petal width is 0.1 cm.
#    - The minimum target value is 0.0.
# 
# 5. **25th Percentile (25%):** This is the value below which 25% of the data falls.
#    - The 25th percentile of sepal length is 5.1 cm.
#    - The 25th percentile of sepal width is 2.8 cm.
#    - The 25th percentile of petal length is 1.6 cm.
#    - The 25th percentile of petal width is 0.3 cm.
#    - The 25th percentile of the target is 0.0.
# 
# 6. **50th Percentile (50% or Median):** This is the median or the value below which 50% of the data falls.
#    - The median sepal length is 5.8 cm.
#    - The median sepal width is 3.0 cm.
#    - The median petal length is 4.35 cm.
#    - The median petal width is 1.3 cm.
#    - The median target is 1.0.
# 
# 7. **75th Percentile (75%):** This is the value below which 75% of the data falls.
#    - The 75th percentile of sepal length is 6.4 cm.
#    - The 75th percentile of sepal width is 3.3 cm.
#    - The 75th percentile of petal length is 5.1 cm.
#    - The 75th percentile of petal width is 1.8 cm.
#    - The 75th percentile of the target is 2.0.
# 
# 8. **Maximum (max):** The maximum values represent the largest observed values for each feature.
#    - The maximum sepal length is 7.9 cm.
#    - The maximum sepal width is 4.4 cm.
#    - The maximum petal length is 6.9 cm.
#    - The maximum petal width is 2.5 cm.
#    - The maximum target is 2.0.
# 
# These statistics give you an overall understanding of the distribution and variability of the dataset. The next steps could involve visualizing the distribution of each feature, exploring relationships between features, and handling any outliers if present. Let me know how you'd like to proceed!
# 
# 
# The new dataset predictions seem to be performing well. The low mean squared error (MSE) of 0.0054 and high R-squared value of 0.9918 suggest that the model is accurately predicting the target variable in the new dataset.
# 
# Here's a brief interpretation of the metrics:
# 
# - **Mean Squared Error (MSE):** This metric represents the average of the squared differences between the actual and predicted values. A lower MSE indicates better model performance.
# 
# - **R-squared (R2):** R-squared is a measure of how well the model's predictions match the actual data. It ranges from 0 to 1, with 1 indicating a perfect fit. In your case, the high R-squared value suggests that the model explains a significant portion of the variance in the target variable.
# 
# 





