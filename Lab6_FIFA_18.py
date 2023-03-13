# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import re
from ydata_profiling import ProfileReport


def compute_operation(cell):
    # Use regular expressions to extract the operation
    match = re.match(r"(\d+)\s*([+-])\s*(\d+)", str(cell))
    if match:
        # If the cell contains an operation, compute the result and return it
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        if op == "+":
            return a + b
        else:
            return a - b
    else:
        # If the cell does not contain an operation, return the original value
        return cell


# Load dataset
data = pd.read_csv("CompleteDataset.csv", low_memory=False)

# Generar un informe detallado de los datos
profile = ProfileReport(data, title="CompleteDataset", explorative=True)

# Guardar el informe en un archivo HTML
profile.to_file("CompleteDataset.html")

# Apply compute_operation function to all columns
data = data.applymap(compute_operation)

# Drop rows with missing values
data = data.dropna()

# Drop unnecessary columns
data.drop(
    [
        "ID",
        "Name",
        "Age",
        "Photo",
        "Nationality",
        "Flag",
        "Club Logo",
        "Value",
        "Wage",
        "Special",
        "Preferred Positions",
        "Number",
        "Club",
    ],
    axis=1,
    inplace=True,
)

data.to_csv("cleaned_data.csv", index=False)

# Split dataset into training, validation, and test sets (80%, 10%, and 10% respectively)
train, valtest = train_test_split(data, test_size=0.2, random_state=1)
val, test = train_test_split(valtest, test_size=0.5, random_state=1)

# Separate features and target variable
train_x = train.drop(["Potential"], axis=1)
train_y = train["Potential"]
val_x = val.drop(["Potential"], axis=1)
val_y = val["Potential"]
test_x = test.drop(["Potential"], axis=1)
test_y = test["Potential"]

# Train Decision Tree model
model = DecisionTreeRegressor(random_state=1)
model.fit(train_x, train_y)

# Get feature importances and sort them in descending order
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Show top 5 features
print("Top 5 features:")
for f in range(5):
    print("%d. %s (%f)" % (f + 1, train_x.columns[indices[f]], importances[indices[f]]))

# Evaluate the model on the validation set
y_val_pred = model.predict(val_x)
print("R2 score on validation set: ", r2_score(val_y, y_val_pred))

# Evaluate the model on the test set
y_test_pred = model.predict(test_x)
print("R2 score on test set: ", r2_score(test_y, y_test_pred))

# Tune hyperparameters using Grid Search
params = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_split": [2, 5, 10, 20],
}
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42), params, cv=5, n_jobs=-1
)
grid_search.fit(pd.concat([train_x, val_x]), pd.concat([train_y, val_y]))

print("Best hyperparameters: ", grid_search.best_params_)

# Predict on test set using best model found by Grid Search
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(test_x)

# Calculate R2 score on test set
print("R2 score on test set using best model: ", r2_score(test_y, y_test_pred))
