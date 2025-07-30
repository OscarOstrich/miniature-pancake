
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

url = 'https://raw.githubusercontent.com/OscarOstrich/miniature-pancake/refs/heads/main/norm_filtered_species_table_MT.csv'

df = pd.read_csv(url)

feature_names = df.iloc[:,0].tolist() #storing the taxa names to be used later for permutation importance

#Transposed the table, now create a final column of CDI+ versus CDI-
def assign_status(sample_id):
    if 'HC' in sample_id:
        return 'CDI+'
    elif 'KR' in sample_id:
        return 'CDI-'
    else:
        return 'Unknown'

status_list = [assign_status(col) for col in df.columns]

# Transpose the dataframe so samples are rows, taxa are columns
csv_T = df.transpose()

csv_T['Status'] = status_list
print(csv_T.head())

x = csv_T.drop('Status', axis=1)
y = csv_T['Status']

y.replace({'CDI+': 1, 'CDI-': 0}, inplace=True)

# Remove rows where Status is 'Unknown'
valid_idx = y[y != 'Unknown'].index

x2 = x.loc[valid_idx]
y2 = y.loc[valid_idx].astype(int)  # Make sure it's integer for classifiers

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

fold_data = []
for train_idx, test_idx in skf.split(x2, y2):
    x_train_fold, x_test_fold = x2.iloc[train_idx], x2.iloc[test_idx]
    y_train_fold, y_test_fold = y2.iloc[train_idx], y2.iloc[test_idx]

    # Convert columns to numeric, coercing errors, and fill NaN
    x_train_fold = x_train_fold.apply(pd.to_numeric, errors='coerce').fillna(0)
    x_test_fold = x_test_fold.apply(pd.to_numeric, errors='coerce').fillna(0)

    fold_data.append((x_train_fold, x_test_fold, y_train_fold, y_test_fold))

""""
FOR MG:
Best Parameters: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100}

Best AUC Score: 0.903125

FOR MT:
Best Parameters: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 200}

Best AUC Score: 0.8794642857142858
"""

#running permutation importance to check weight and contribution of each feature
bParams2_model = RandomForestClassifier(n_estimators=200,
                              max_depth=5,
                              max_features='sqrt',
                              min_samples_split=5,
                              random_state=123
)

# Fit the model before calculating permutation importance
bParams2_model.fit(x_train_fold, y_train_fold)
y_pred = bParams2_model.predict(x_test_fold)

print(accuracy_score(y_test_fold,y_pred))
print(recall_score(y_test_fold,y_pred))
print(precision_score(y_test_fold,y_pred))

result = permutation_importance(
    bParams2_model, x_test_fold, y_test_fold,
    scoring='roc_auc', n_repeats=5, random_state=123, n_jobs=-1
)

importances_mean = result.importances_mean
importances_std = result.importances_std

# Get indices of features with non-zero importance
non_zero_idx = np.where(importances_mean > 0)[0]

# Sort them in descending order
sorted_idx = non_zero_idx[np.argsort(importances_mean[non_zero_idx])[::-1]]

# Limit to top 20
top_n = min(20, len(sorted_idx))
sorted_idx = sorted_idx[:top_n]

# Get corresponding names
sorted_feature_names = [feature_names[i] for i in sorted_idx]

plt.figure(figsize=(14, 10))  # wider figure
plt.barh(
    y=np.arange(len(sorted_idx)),
    width=importances_mean[sorted_idx],
    xerr=importances_std[sorted_idx],
    align='center',
    color='skyblue',
    edgecolor='black'
)

# Fix the y-axis labels
plt.yticks(
    ticks=np.arange(len(sorted_idx)),
    labels=sorted_feature_names,
    fontsize=8  
)

plt.xlabel("Mean decrease in AUC (importance)", fontsize=12)
plt.title("Permutation Importance - Top Features", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

