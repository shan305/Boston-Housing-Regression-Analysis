# Machine Learning Project README

## Overview

This machine learning project involves building and evaluating regression models using the Boston Housing Dataset. The code is modularized to enhance readability and maintainability, with each module focusing on a specific aspect of the machine learning pipeline.

## Project Structure

1. **`load_data.py`**: Loads the Boston Housing Dataset and performs initial data processing.

2. **`explore_visualize.py`**: Conducts exploratory data analysis (EDA) and visualizes the correlation matrix of the features.

3. **`prepare_data.py`**: Prepares the data for regression by splitting it into training and testing sets.

4. **`train_models.py`**: Trains linear regression, decision tree regressor, and random forest regressor models.

5. **`evaluate_models.py`**: Evaluates the performance of the trained models on the test set.

6. **`visualize_predictions.py`**: Visualizes the predictions of the trained models for comparison.

7. **`fine_tune_random_forest.py`**: Performs hyperparameter tuning for the random forest regressor using GridSearchCV.

8. **`save_model.py`**: Saves the best random forest regressor model to a file.

9. **`load_and_predict.py`**: Loads the saved model and makes predictions on new data.

10. **`feature_importance.py`**: Visualizes the feature importance plot for the best random forest regressor.

## How to Use

1. **Install Dependencies:**
   - Ensure you have Python installed on your machine.
   - Install required libraries by running `pip install -r requirements.txt`.

2. **Run the Code:**
   - Execute the modularized code by running `python main.py`.

3. **Review Results:**
   - Check the console for model evaluation metrics such as Mean Squared Error and R-squared.
   - Visualizations of predictions and feature importance plots will be displayed.

4. **Save Model for Future Use:**
   - The best random forest regressor model is saved to a file (`best_random_forest_model.joblib`) for future use.

5. **Load and Predict on New Dataset:**
   - Use the loaded model to make predictions on a new dataset.

## Additional Notes

- The project structure is designed to promote code modularity and organization.
- Each module corresponds to a specific step in the machine learning pipeline, facilitating ease of understanding and maintenance.
- Feel free to customize and extend the code according to your specific requirements.

Happy coding!
