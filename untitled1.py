from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np
import pandas as pd

# Load the uploaded CSV file to inspect its contents
data = pd.read_csv("data.csv")

# Display the first few rows of the dataset
data.head()
# Data preprocessing
# Dropping rows with missing values in relevant columns
cleaned_data = data.dropna(subset=['averageRating', 'numVotes', 'releaseYear'])

# Define independent variables (features) and dependent variable (target)
X = cleaned_data[['numVotes', 'releaseYear']]
y_continuous = cleaned_data['averageRating']

# For logistic regression, we categorize ratings into "high" (>= 6.0) and "low" (< 6.0)
y_categorical = (cleaned_data['averageRating'] >= 6.0).astype(int)

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train_cont, y_test_cont = train_test_split(X, y_continuous, test_size=0.2, random_state=42)
_, _, y_train_cat, y_test_cat = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train_cont)
y_pred_cont = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test_cont, y_pred_cont)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train_cat)
y_pred_cat = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test_cat, y_pred_cat)

print("Linear Regression =",linear_mse)
print("Logistic Regression =",logistic_accuracy*100,"%")
