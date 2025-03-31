import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

# Collecting Training and Testing Dataset File Paths

dfps_train = []
dfps_test = []

for dirname, _, filenames in os.walk("./cicddos2019/"):
    for filename in filenames:
        if filename.endswith("-training.parquet"):
            dfp = os.path.join(dirname, filename)
            dfps_train.append(dfp)
            print(dfp)
        elif filename.endswith("-testing.parquet"):
            dfp = os.path.join(dirname, filename)
            dfps_test.append(dfp)
            print(dfp)

# Common Prefixes in both lists
train_prefixes = [dfp.split("/")[-1].split("-")[0] for dfp in dfps_train]
test_prefixes = [dfp.split("/")[-1].split("-")[0] for dfp in dfps_test]

common_prefixes = list(set(train_prefixes).intersection(test_prefixes))

# Filter the dataframes to only include the common prefixes
dfps_train = [dfp for dfp in dfps_train if dfp.split("/")[-1].split("-")[0] in common_prefixes]
dfps_test = [dfp for dfp in dfps_test if dfp.split("/")[-1].split("-")[0] in common_prefixes]


train_df = pd.concat([pd.read_parquet(dfp) for dfp in dfps_train], ignore_index=True)
test_df = pd.concat([pd.read_parquet(dfp) for dfp in dfps_test], ignore_index=True)

# Drop the WebDDoS class from the testing data
test_df = test_df[test_df["Label"] != "WebDDoS"]

# Map the labels to the same format
label_mapping = {
    "DrDoS_UDP": "UDP",
    "UDP-lag": "UDPLag",
    "DrDoS_MSSQL": "MSSQL",
    "DrDoS_LDAP": "LDAP",
    "DrDoS_NetBIOS": "NetBIOS",
    "Syn": "Syn",  # Already matches
    "Benign": "Benign",  # Already matches
}

test_df["Label"] = test_df["Label"].map(label_mapping)

train_df = train_df[~train_df["Label"].isin(["NetBIOS", "UDPLag"])]
test_df = test_df[~test_df["Label"].isin(["NetBIOS", "UDPLag"])]

# Features with a single unique value
single_val_cols = [col for col in train_df.columns if train_df[col].nunique() == 1]

# Remove columns with a single unique value
train_df.drop(single_val_cols, axis=1, inplace=True)
test_df.drop(single_val_cols, axis=1, inplace=True)


def grab_col_names(data, cat_th=10, car_th=20):
    # Categorical columns and categorical but high-cardinality columns
    cat_cols = [col for col in data.columns if data[col].dtypes == "O"]
    num_but_cat = [col for col in data.columns if data[col].nunique() < cat_th and data[col].dtypes != "O"]
    high_card_cat_cols = [col for col in data.columns if data[col].nunique() > car_th and data[col].dtypes == "O"]

    # Combine Object type columns and Low-unique-value numeric columns into cat_cols
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in high_card_cat_cols]

    # Numerical columns excluding those considered as categorical
    num_cols = [col for col in data.columns if data[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Display information about the dataset
    print(f"Observations: {data.shape[0]}")
    print(f"Variables: {data.shape[1]}")
    print(f"Categorical Columns: {len(cat_cols)}")
    print(f"Numerical Columns: {len(num_cols)}")
    print(f"High Cardinality Categorical Columns: {len(high_card_cat_cols)}")
    print(f"Number but Categorical Columns: {len(num_but_cat)}")
    print("\n")

    return cat_cols, num_cols, high_card_cat_cols


cat_cols, num_cols, high_card_cat_cols = grab_col_names(train_df)
print(f"Categorical Columns: {cat_cols}")
print(f"Numerical Columns: {num_cols}")
print(f"High Cardinality Categorical Columns: {high_card_cat_cols}")

# Remove duplicate rows
train_df = train_df.drop_duplicates()

# Select only numeric columns
numerical_df = train_df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numerical_df.corr().abs()

# Generate a boolean mask for the upper triangle
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

# Select the upper triangle of the correlation matrix
upper_triangle = corr_matrix.where(mask)

# Find the columns with correlation of 0.8 or higher
high_corr_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.8)]

# Display the number of highly correlated columns and their names
print(f"Total number of highly correlated columns: {len(high_corr_cols)}")
print("Highly correlated columns are:", high_corr_cols)

# Remove highly correlated columns from the dataset
train_df.drop(high_corr_cols, axis=1, inplace=True)
test_df.drop(high_corr_cols, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    train_df.drop("Label", axis=1), train_df["Label"], test_size=0.2, random_state=42
)
X_val, y_val = test_df.drop("Label", axis=1), test_df["Label"]

# Encode the target variable

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

# Label mapping for the target variable
label_map = {index: Label for index, Label in enumerate(le.classes_)}

# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
X_train_encoded = scaler.fit_transform(X_train)
X_val_encoded = scaler.transform(X_val)
X_test_encoded = scaler.transform(X_test)

model = XGBClassifier(
    eval_metric="logloss",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train_encoded, y_train)
y_val_pred = model.predict(X_val_encoded)
y_test_pred = model.predict(X_test_encoded)

# Evaluate predictions for validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
val_recall = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
val_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
# Evaluate predictions for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

# Display the performance metrics
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# # Get feature importances
importance = model.feature_importances_

# We need to use the column names from before encoding
feature_names = X_train.columns.tolist()

# Create a dataframe of feature names and their importance scores
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False)

# Display top 20 most important features
print("\nTop 20 Most Important Features:")
print(feature_importance_df.head(20))

# Optionally, visualize with a bar plot
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df["Feature"][:20], feature_importance_df["Importance"][:20])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importance")
plt.gca().invert_yaxis()  # To have the highest importance at the top
plt.tight_layout()
plt.savefig("cic_feature_importance.png")
plt.show()


# Get original class names from the label encoder
class_names = le.classes_

# Create confusion matrix for test data
cm = confusion_matrix(y_test, y_test_pred)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("cic_confusion_matrix.png")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))
