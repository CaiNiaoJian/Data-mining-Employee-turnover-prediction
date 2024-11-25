import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the result files
result = pd.read_csv('result.csv', index_col='ID')
analysis_result = pd.read_csv('analysis_result.csv', index_col='ID')

# Ensure the files have the same structure and IDs
if not result.index.equals(analysis_result.index):
    raise ValueError("The IDs in the two files do not match.")

# Compare the predictions
result_comparison = pd.DataFrame({
    'Actual': result['Label'],
    'Predicted': analysis_result['Label']
})

# Calculate similarity metrics
accuracy = accuracy_score(result_comparison['Actual'], result_comparison['Predicted'])
conf_matrix = confusion_matrix(result_comparison['Actual'], result_comparison['Predicted'])

# Print accuracy
print(f"Similarity (Accuracy): {accuracy:.2f}")

# Calculate attrition rates
result_attrition_rate = result['Label'].mean()
analysis_attrition_rate = analysis_result['Label'].mean()

print(f"Result dataset attrition rate: {result_attrition_rate:.2%}")
print(f"Analysis result dataset attrition rate: {analysis_attrition_rate:.2%}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Left', 'Left'], yticklabels=['Not Left', 'Left'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot reliability diagram
result_comparison['Correct'] = result_comparison['Actual'] == result_comparison['Predicted']
grouped = result_comparison.groupby('Actual')['Correct'].mean()

plt.figure(figsize=(8, 6))
plt.bar(grouped.index, grouped.values, tick_label=['Not Left', 'Left'], color=['skyblue', 'salmon'])
plt.ylabel('Proportion Correct')
plt.title('Prediction Reliability by Class')
plt.ylim(0, 1)
plt.show()

print("Comparison and visualization complete.")
