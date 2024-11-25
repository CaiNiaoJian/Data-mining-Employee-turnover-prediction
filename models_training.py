import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the cleaned training dataset
train = pd.read_csv('cleaned_train.csv', index_col='ID')

# Debugging: Print column names to verify data
print("cleaned_train.csv 中的列名:", train.columns)

# Split data into features and labels
X = train

# Simulate labels for training (assuming binary classification for employee attrition)
# NOTE: Replace this with actual labels when available
y = (train.index % 2).astype(int)  # Temporary simulated labels

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a RandomForestClassifier model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model on the validation set
y_pred = best_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Load the cleaned test dataset
test = pd.read_csv('cleaned_test.csv', index_col='ID')

# Predict on the test dataset
test_predictions = best_model.predict(test)

# Prepare the output DataFrame with ID and predicted labels
output = pd.DataFrame({'ID': test.index, 'Label': test_predictions})

# Save the predictions to a CSV file
output.to_csv('analysis_result.csv', index=False)

print("Model training complete. Predictions saved to 'analysis_result.csv'.")
