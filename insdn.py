import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

csv_files = os.listdir("./InSDN")
df = pd.DataFrame()

li = []
for filename in csv_files:
    df = pd.read_csv(os.path.join("./InSDN", filename), low_memory=False, index_col=None, header=0)
    li.append(df)
    print("Read in {}".format(filename))

df = pd.concat(li, axis=0, ignore_index=True)
print("Finished reading in {} entire".format(str(df.shape[0])))

df["Label"] = df["Label"].str.strip().str.lower()

# Features with a single unique value
single_val_cols = [col for col in df.columns if df[col].nunique() == 1]

# Remove columns with a single unique value
df.drop(single_val_cols, axis=1, inplace=True)


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


cat_cols, num_cols, high_card_cat_cols = grab_col_names(df)
print(f"Categorical Columns: {cat_cols}")
print(f"Numerical Columns: {num_cols}")
print(f"High Cardinality Categorical Columns: {high_card_cat_cols}")

df = df.drop_duplicates()

# Select only numeric columns
numerical_df = df.select_dtypes(include=[np.number])

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
df.drop(high_corr_cols, axis=1, inplace=True)
df.drop(high_card_cat_cols, axis=1, inplace=True)

X = df.drop("Label", axis=1)
y = df["Label"]

# First split: 70% training and 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Second split: 50% validation and 50% test from the temporary set
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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
# importance = model.feature_importances_

# # Get feature names from original dataset
# # We need to use the column names from before encoding
feature_names = X_train.columns.tolist()

# # Create a dataframe of feature names and their importance scores
# feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})

# # Sort by importance
# feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False)

# # Display top 20 most important features
# print("\nTop 20 Most Important Features:")
# print(feature_importance_df.head(20))

# # Optionally, visualize with a bar plot
# plt.figure(figsize=(12, 8))
# plt.barh(feature_importance_df["Feature"][:20], feature_importance_df["Importance"][:20])
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.title("Top 20 Feature Importance")
# plt.gca().invert_yaxis()  # To have the highest importance at the top
# plt.tight_layout()
# plt.savefig("insdn_feature_importance.png")
# plt.show()


# # Get original class names from the label encoder
# class_names = le.classes_

# # Create confusion matrix for test data
# cm = confusion_matrix(y_test, y_test_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("insdn_confusion_matrix.png")
# plt.show()

# print("\nClassification Report:")
# print(classification_report(y_test, y_test_pred, target_names=class_names))


# Function to display top important features for a class
def display_top_features_for_class(class_index, class_name, n_top=10):
    print(f"\nTop {n_top} features for class: {class_name}")

    # Create a binary classification problem (one-vs-rest)
    y_train_binary = np.where(y_train == class_index, 1, 0)

    # Train a new model specifically for this class
    class_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    class_model.fit(X_train_encoded, y_train_binary)

    # Get feature importances for this specific class
    importance = class_model.feature_importances_

    # Create a dataframe of feature names and their importance scores
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False)

    # Display top N features
    print(feature_importance_df.head(n_top))

    return feature_importance_df


def analyze_permutation_importance(class_index, class_name, n_top=10):
    print(f"\nPermutation Importance for class: {class_name}")

    # Create a binary classification problem (one-vs-rest)
    y_test_binary = np.where(y_test == class_index, 1, 0)

    # Train a new model specifically for this class
    class_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    class_model.fit(X_train_encoded, np.where(y_train == class_index, 1, 0))

    # Calculate permutation importance on test set
    perm_importance = permutation_importance(
        class_model, X_test_encoded, y_test_binary, n_repeats=10, random_state=42, n_jobs=-1
    )

    # Create a dataframe with feature importance
    perm_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": perm_importance.importances_mean})

    # Sort by importance
    perm_importance_df = perm_importance_df.sort_values("Importance", ascending=False)

    # Display top N features
    print(perm_importance_df.head(n_top))

    return perm_importance_df


# Get feature importance for each class
class_importance_results = {}
permutation_importance_results = {}

print("\n=== Per-Class Feature Importance Analysis ===")

# Iterate through each class
for class_index, class_name in enumerate(le.classes_):
    print(f"\n{'=' * 50}")
    print(f"Analyzing class: {class_name} (index: {class_index})")

    # Get important features using direct model training
    class_importance_results[class_name] = display_top_features_for_class(class_index, class_name)

    # Get important features using permutation importance
    permutation_importance_results[class_name] = analyze_permutation_importance(class_index, class_name)

plt.figure(figsize=(15, len(le.classes_) * 3))

for i, class_name in enumerate(le.classes_):
    # Get the top 5 features for this class
    top_features = permutation_importance_results[class_name].head(5)

    # Create subplot
    plt.subplot(len(le.classes_), 1, i + 1)
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title(f"Top 5 features for {class_name}")
    plt.tight_layout()

plt.savefig("insdn_per_class_feature_importance_2.png")
plt.show()
