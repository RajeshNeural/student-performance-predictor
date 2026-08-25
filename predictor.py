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

print("\n===================================")
print("   STUDENT PERFORMANCE PREDICTOR")
print("===================================")

print(f"\nModel MAE : {mae:.2f}")
print(f"Model R²  : {r2:.2f}")

# Get user input
print("\nEnter student details:")
print("-----------------------------------")

study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
previous_score = float(input("Previous exam score: "))
sleep_hours = float(input("Sleep hours per day: "))
assignments_completed = int(input("Assignments completed: "))

# Create input data
student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "sleep_hours": [sleep_hours],
    "assignments_completed": [assignments_completed]
})

# Predict final score
predicted_score = model.predict(student)[0]

print("\n===================================")
print(f" Predicted Final Score: {predicted_score:.2f}")
print("===================================\n")
