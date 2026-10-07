import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("part1_return_prediction/data/orders_dataset.csv")

# Remove columns that should not be used for prediction
df = df.drop(columns=["order_id", "rating_given"])

# Separate features and target
X = df.drop(columns=["returned"])
y = df["returned"]

# print("\n--- Features ---")
# print(X.columns.tolist())

# print("\n--- Target ---")
# print(y.name)

# print("\n--- X Shape ---")
# print(X.shape)

# print("\n--- y Shape ---")
# print(y.shape)

# print("\n--- Missing Values ---")
# print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# print("\n--- Train/Test Shape ---")
# print("X_train:", X_train.shape)
# print("X_test :", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test :", y_test.shape)

# Categorical and numerical columns
categorical_features = [
    "product_category",
    "payment_method"
]

numerical_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features),
        ("numerical","passthrough",numerical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print("\n--- Processed Data Shape ---")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed :", X_test_processed.shape)