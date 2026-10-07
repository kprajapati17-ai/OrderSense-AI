import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv(
    "part1_return_prediction/data/orders_dataset.csv"
)


# --------------------------------------------------
# 2. Remove Columns Not Used for Prediction
# --------------------------------------------------

df = df.drop(columns=["order_id", "rating_given"])


# --------------------------------------------------
# 3. Separate Features and Target
# --------------------------------------------------

X = df.drop(columns=["returned"])
y = df["returned"]


# --------------------------------------------------
# 4. Feature Engineering
# --------------------------------------------------

# Previous return rate
X["previous_return_rate"] = (
    X["num_previous_returns"]
    / X["num_previous_orders"].replace(0, 1)
)


# --------------------------------------------------
# 5. Define Categorical Features
# --------------------------------------------------

categorical_features = [
    "product_category",
    "payment_method"
]


# --------------------------------------------------
# 6. Define Numerical Features
# --------------------------------------------------

numerical_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "previous_return_rate"
]


# --------------------------------------------------
# 7. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 8. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n--- Dataset Split ---")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# 9. Random Forest Model
# --------------------------------------------------

model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)


# --------------------------------------------------
# 10. Hyperparameter Search Space
# --------------------------------------------------

param_distributions = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}


# --------------------------------------------------
# 11. Complete ML Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 12. Randomized Hyperparameter Search
# --------------------------------------------------

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 13. Train and Tune Model
# --------------------------------------------------

print("\n--- Hyperparameter Tuning Started ---")

search.fit(X_train, y_train)

print("\n--- Hyperparameter Tuning Completed ---")


# --------------------------------------------------
# 14. Best Parameters
# --------------------------------------------------

print("\n--- Best Parameters ---")
print(search.best_params_)


# --------------------------------------------------
# 15. Best Cross-Validation F1 Score
# --------------------------------------------------

print("\n--- Best Cross-Validation F1 ---")
print(search.best_score_)


# --------------------------------------------------
# 16. Prediction on Unseen Test Data
# --------------------------------------------------

y_pred = search.predict(X_test)


# --------------------------------------------------
# 17. Model Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n--- Model Evaluation ---")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# --------------------------------------------------
# 18. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)