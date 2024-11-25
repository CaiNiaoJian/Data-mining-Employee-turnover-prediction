import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load train and test data
train = pd.read_csv('Sample_train.csv', index_col='ID')
test = pd.read_csv('test_noLabel.csv', index_col='ID')

# Function to drop useless attributes
def drop_useless_attribution(data):
    data.drop(['EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace=True)

drop_useless_attribution(train)
drop_useless_attribution(test)

# Map ordinal categorical attributes
def map_categorical_attributes(data):
    data['BusinessTravel'] = data['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    data['OverTime'] = data['OverTime'].map({'No': 0, 'Yes': 1})

map_categorical_attributes(train)
map_categorical_attributes(test)

# Drop irrelevant categorical attributes
train.drop('Gender', axis=1, inplace=True)
test.drop('Gender', axis=1, inplace=True)

# One-hot encode nominal categorical attributes
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Add new attributes
def add_new_attributes(data):
    data['JobInvolvementPerPercentSalaryHike'] = data['PercentSalaryHike'] / data['JobInvolvement']
    data['JobInvolvementSatisfaction'] = data['JobSatisfaction'] / data['JobInvolvement']
    data['MaritalStatus_MarriedOverTime'] = np.where(data['MaritalStatus_Married'] == 1, data['OverTime'], 0)
    data['MaritalStatus_MarriedBusinessTravel'] = np.where(data['MaritalStatus_Married'] == 1, data['BusinessTravel'], 0)
    data['AverageWorkYearsPerCompany'] = data['TotalWorkingYears'] / (data['NumCompaniesWorked'] + 1)

add_new_attributes(train)
add_new_attributes(test)

# Identify highly correlated attribute pairs
corr = train.corr()
corr_where = np.where(corr[corr.abs() < 1].abs() >= 0.8)
corr_att_list = [(corr_where[0][i], corr_where[1][i]) for i in range(len(corr_where[0]))]

# Drop highly correlated attributes
def drop_highly_correlated_attributes(data):
    data.drop(['MonthlyIncome', 'JobRole_Human Resources', 'JobRole_Sales Executive'], axis=1, inplace=True)

drop_highly_correlated_attributes(train)
drop_highly_correlated_attributes(test)

# Normalize ratio attributes with Z-score normalization
def normalize_ratio_attributes(data):
    ratio_cols = [0, 2, 8, 10, 14, 17, 18, 19, 20, 40, 41, 44]
    data.iloc[:, ratio_cols] = StandardScaler().fit_transform(data.iloc[:, ratio_cols])

normalize_ratio_attributes(train)
normalize_ratio_attributes(test)

# Normalize ordinal attributes with Min-Max scaling
def normalize_ordinal_attributes(data):
    ordinal_cols = [1, 3, 4, 5, 6, 7, 11, 12, 13, 15, 16, 42, 43]
    data.iloc[:, ordinal_cols] = MinMaxScaler().fit_transform(data.iloc[:, ordinal_cols])

normalize_ordinal_attributes(train)
normalize_ordinal_attributes(test)

# Extract labels from training data
train_Label = train.pop('Label')
train_Label = pd.DataFrame(train_Label)

# Additional processing steps
# Fill missing values (if any) with median values
def fill_missing_values(data):
    data.fillna(data.median(), inplace=True)

fill_missing_values(train)
fill_missing_values(test)

# Save cleaned data to CSV
train.to_csv('cleaned_train.csv', index=True)
test.to_csv('cleaned_test.csv', index=True)

# Save the structure analysis result to a text file
with open('FEDemoresult.txt', 'w') as file:
    buffer = []
    train.info(verbose=True, buf=buffer)
    file.write("\n".join(buffer))

print("Data processing complete. Cleaned datasets saved as 'cleaned_train.csv' and 'cleaned_test.csv'. Structure analysis saved to FEDemoresult.txt.")
