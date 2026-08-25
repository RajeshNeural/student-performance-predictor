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


# Safe input function
def get_number(message, minimum, maximum):
    while True:
        try:
            value = float(input(message))

            if minimum <= value <= maximum:
                return value

            print(f"Please enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Invalid input. Please enter a number.")


# Get user input
print("\nEnter student details:")
print("-----------------------------------")

study_hours = get_number(
    "Study hours per day (0-24): ", 0, 24
)

attendance = get_number(
    "Attendance percentage (0-100): ", 0, 100
)

previous_score = get_number(
    "Previous exam score (0-100): ", 0, 100
)

sleep_hours = get_number(
    "Sleep hours per day (0-24): ", 0, 24
)

assignments_completed = get_number(
    "Assignments completed (0-100): ", 0, 100
)


# Create student data
student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "sleep_hours": [sleep_hours],
    "assignments_completed": [assignments_completed]
})


# Predict final score
predicted_score = model.predict(student)[0]

# Keep prediction between 0 and 100
predicted_score = max(0, min(100, predicted_score))


print("\n===================================")
print(f" Predicted Final Score: {predicted_score:.2f}")
print("===================================\n")
