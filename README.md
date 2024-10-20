# Simple-Linear-Regression-Housing-Prediction-

House Price Prediction Project

Overview
This project demonstrates how to build a simple linear regression model in Python to predict house prices using a dataset from Kaggle. 
The project covers data preprocessing, feature selection, model training, evaluation, and cross-validation, with a focus on educational insights into each step.

Files
- house_prices_data: Contains the raw data files.
- main.py: Contains the code for model training and evaluation.
- README.md: Project documentation.



Step-by-Step Process:

1. Data Preprocessing
Preprocessing is a critical step before building any machine learning model. Here's how I handled the dataset:
- Handling Missing Values: I filled missing values in the LotFrontage column using the mean value to avoid data loss during training.

- Scaling Numeric Data: Used StandardScaler to normalize numeric columns, ensuring all features were on a similar scale for better model performance.

- One-Hot Encoding Categorical Variables: Converted categorical variables (like Neighborhood or HouseStyle) into numerical form using one-hot encoding to make them usable by the regression model.




2. Feature Selection
I used a correlation heatmap to identify which features were most strongly correlated with the target variable (`SalePrice`). By focusing on features with a correlation above 0.6,  the following key features were selected:
- GrLivArea (Living area square footage) with a correlation of 0.71.
- TotalBsmtSF (Total basement square footage) with a correlation of 0.61.
- OverallQual (Overall material and finish quality) with a correlation of 0.79.


3. Model Training
For this project, we built a Linear Regression model using the selected features and split the data into training and testing sets to evaluate performance.

- Training : 80% of the data was used for training the model, and 20% was reserved for testing.
- Linear Regression : Chosen for its simplicity and interpretability.


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


4. Evaluation

I used Mean Squared Error (MSE) to evaluate the model’s performance, and implemented  5-fold cross-validation to ensure the model generalized well across different subsets of the data.
MSE: Measures the average squared difference between predicted and actual values.
Cross-validation: Splits the data into 5 subsets, trains on 4, and tests on 1 in a rotating fashion, providing a more reliable measure of model performance.

Results:
- MSE: 1.67 billion
- Cross-validated MSE: 1.69 billion


5. Visualization

We created several visualizations to better understand the relationships in the data and model performance:
- Correlation Heatmap: Visualized the correlation between features and `SalePrice` to aid in feature selection.
- Actual vs. Predicted Plot: A scatter plot to compare actual house prices with the predicted values.

Code example:
python
import matplotlib.pyplot as plt

# Actual vs. Predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.show()



Key Learnings
- Data Preprocessing: Handling missing values and normalizing data are essential to ensure model reliability.
- Feature Selection: Using a heatmap helps identify which features strongly influence the target variable.
- Model Evaluation: Cross-validation provides a more robust evaluation of model performance compared to a single train-test split.
- Visualization: Plotting results helps in better understanding the model’s strengths and weaknesses.


Results
- Cross-Validated MSE: 1.69 billion  
The model performs consistently, with no significant signs of overfitting, as indicated by the similar results between the regular MSE and cross-validated MSE.
