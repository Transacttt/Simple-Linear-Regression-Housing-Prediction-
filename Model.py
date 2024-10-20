from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

from sklearn.preprocessing import StandardScaler




### Step 1: Data Collection via Kaggle API (Instead of CLI)


# Initialize and authenticate the API
api = KaggleApi()
api.authenticate()

# Download the dataset
competition_name = 'house-prices-advanced-regression-techniques'     
api.competition_download_files(competition_name, path='house_prices_data') #creates file called house_prices_data and saves the corresponding files to it

# Unzip the dataset
with zipfile.ZipFile('house_prices_data/house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
    zip_ref.extractall('house_prices_data')

# Load train and test data
import pandas as pd
train_data = pd.read_csv('house_prices_data/train.csv')
test_data = pd.read_csv('house_prices_data/test.csv')

#print(train_data.head())






### Step 2: Data Preprocessing

#PRE PROCESSING TIMEEE

# Handle missing values (e.g., filling with mean)
# Fill missing values only for numerical columns
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
#train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())


#DATA normalisation


# Select numerical columns
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns

# Scale only numerical columns
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data[numeric_cols])
# One-hot encode categorical columns

#print(train_data.columns) (used this for testing stuff)

sale_price = train_data['SalePrice']


#train_data = pd.get_dummies(train_data)  #  converts categorical variables into numerical ones by creating dummy/indicator variables (one-hot encoding)

# Fill missing values in LotFrontage with the mean (or another appropriate value)
train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean(), inplace=True)

# Then recalculate the correlation and generate the heatmap

train_data = pd.get_dummies(train_data.drop('SalePrice', axis=1))


train_data['SalePrice'] = sale_price


# Calculate the correlation matrix
correlation_matrix = train_data.corr()

# Print the correlation matrix to check if SalePrice and LotFrontage are included
#print(correlation_matrix)  #<---- confirms stuff is being sized out in the heatmap





#print(train_data.isnull().sum())  # Check for missing values in each column
#print(train_data.isna().sum())    # Check for NaN values in each column




#print(train_data.columns) (used this for testing stuff)


### Step 3: Exploratory Data Analysis (EDA)

#I'm using a heatmap for this part of the project.
#Issue I encountered was when analysing my heatmap I couldn't find the sale price... Maybe my heatmap is dealing with the test data? and not the training?
#Nah during one hot encoding it makes mistakes confusing sale price for a categorical I'll just seperate it (PROBLEM SOLVED TICK)


#ANOTHER ISSUE not all features are appearing on both axes along the correlation matrix. When I zoom in the red line doesn't show it on both sides
#I'll check then number of null and not a number values.

#Thought: I think the labels ar all present now and some are just invisible due to sizing issue I'll check but I'm getting mentally fatigued right now so.. break time




#import matplotlib.pyplot as plt
#import seaborn as sns

# Correlation heatmap
#plt.figure(figsize=(20, 16))
#sns.heatmap(train_data.corr(),annot=True, cmap="coolwarm",annot_kws={"size": 6})
#plt.xticks(rotation=90)  # Rotate x-axis labels to avoid overlap
#plt.yticks(rotation=0)   # Keep y-axis labels horizontal
#plt.show()



## Sizing error alright? it's confirmed and it's fixed. 
## Heatmap created and done with no more issues that I can see.


##My analysis of the heatmap:
## There are extra features included in the heatmap graph that bare some correlation towards the "SalePrice" which is why some SPACES have none white non zero correlation 
#values. Due to one hot encoding I'd wager. I won't use them as part of my predictive model feels too long. Wanna get this done.
## Labels that have strong correlation (correlation above 0.6) to Saleprice that are labelled and visible:
# 1) GrLiveArea(0.71) 2)TotalBsmtSF (0.61) 3)OverallQual(0.79)
#I think I'll just proceed with these



#So the features selected : 1) GrLiveArea(0.71) 2)TotalBsmtSF (0.61) 3)OverallQual(0.79)


# Define features (selected from heatmap analysis)
features = ['GrLivArea', 'TotalBsmtSF', 'OverallQual']

# Create feature matrix X and target vector y
X = train_data[features]
y = train_data['SalePrice']


# Split data into training and testing sets (80% train, 20% test)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Train the Model (Linear Regression)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model

mean_saleprice = train_data['SalePrice'].mean()
std_saleprice = train_data['SalePrice'].std()


#A good MSE would be around or below the standard deviation squared (std_saleprice ** 2), but the lower, the better
print (std_saleprice**2)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


##Cross Validation for my model
from sklearn.model_selection import cross_val_score

# Cross-validate using 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean MSE across the 5 folds
mse_scores = -scores
print(f"Cross-validated MSE: {mse_scores.mean()}")
#Mean Squared Error: 1667657527.1633685
#Cross-validated MSE: 1695244716.565646  these numbers indicate consistent results not major signs of overfitting yeah I think the models alright.


##Visualisations
import matplotlib.pyplot as plt

# Plot predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.show()

# Plot residuals (errors) to see how well the model fits
residuals = y_test - y_pred
plt.hist(residuals, bins=50)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()

#Distribution of residuals are looking normal pretty good.
