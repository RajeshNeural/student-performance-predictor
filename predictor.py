import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "sleep_hours",
        "assignments_completed"
    ]
]

y = data["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Performance Predictor")
print("-----------------------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Example prediction
student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [90],
    "previous_score": [78],
    "sleep_hours": [7],
    "assignments_completed": [9]
})

predicted_score = model.predict(student)[0]

print(f"\nPredicted Final Score: {predicted_score:.2f}")
