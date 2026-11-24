from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("student_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignment_marks = float(request.form["assignment_marks"])
    sleep_hours = float(request.form["sleep_hours"])

    # Input validation
    if not (0 <= attendance <= 100):
        return "Attendance must be between 0 and 100."

    if not (0 <= previous_marks <= 100):
        return "Previous marks must be between 0 and 100."

    if not (0 <= assignment_marks <= 100):
        return "Assignment marks must be between 0 and 100."

    if not (0 <= study_hours <= 24):
        return "Study hours must be between 0 and 24."

    if not (0 <= sleep_hours <= 24):
        return "Sleep hours must be between 0 and 24."

    # Prepare input for prediction
    input_data = [[
        study_hours,
        attendance,
        previous_marks,
        assignment_marks,
        sleep_hours
    ]]

    # Predict student marks
    prediction = model.predict(input_data)[0]

    # Keep predicted marks between 0 and 100
    prediction = max(0, min(100, prediction))

    prediction = round(prediction, 2)

    # Determine performance category
    if prediction >= 85:
        feedback = "Excellent performance! Keep maintaining your study habits"
    elif prediction >= 70:
          feedback = "Very Good performance! Keep practicing and revise regularly"
    elif prediction >= 60:
          feedback = "Good performance! Focus more on difficult subjects"
    elif prediction >= 50:
          feedback = "Average performance! Try improving study consistency"
    else:
         feedback = "Needs Improvement! Make a study plan and practice regularly"

    return render_template(
        "index.html",
        prediction=prediction,
        performance=feedback,
        feedback=feedback
    )


if __name__ == "__main__":
    app.run(debug=True)
    