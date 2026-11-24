import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv("dataset.csv")

# Select input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_marks",
        "sleep_hours",
    ]
]

# Select target variable
y = data["final_marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Machine Learning model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Performance Prediction Model")
print("----------------------------------")
print("Model training completed successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Train the final model on the complete dataset
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_model.pkl")

print("\nModel saved successfully as student_model.pkl")