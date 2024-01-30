import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('train.csv')

# Use the 'GrLivArea' as the feature and 'SalePrice' as the target
X = df[['GrLivArea']].values
y = df['SalePrice'].values

# Number of folds for cross-validation
num_folds = 10

# Initialize a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store accuracy for each fold
accuracies = []

# Loop through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    accuracy = 1 - (mse / np.var(y_test))

    # Store accuracy for this fold
    accuracies.append(accuracy)

    # Optionally, print the accuracy for each fold
    print(f'Fold Accuracy: {accuracy}')

# Calculate overall accuracy
overall_accuracy = np.mean(accuracies)
print(f'Overall Accuracy: {overall_accuracy}')
