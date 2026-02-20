from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("random_forest_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        age_first_funding_year = float(request.form["age_first_funding_year"])
        age_last_funding_year = float(request.form["age_last_funding_year"])
        age_first_milestone_year = float(request.form["age_first_milestone_year"])
        age_last_milestone_year = float(request.form["age_last_milestone_year"])
        relationships = float(request.form["relationships"])
        funding_rounds = float(request.form["funding_rounds"])
        funding_total_usd = float(request.form["funding_total_usd"])
        milestones = float(request.form["milestones"])
        avg_participants = float(request.form["avg_participants"])
        is_top500 = float(request.form["is_top500"])

        # Arrange features in same order as training
        features = [[
            age_first_funding_year,
            age_last_funding_year,
            age_first_milestone_year,
            age_last_milestone_year,
            relationships,
            funding_rounds,
            funding_total_usd,
            milestones,
            avg_participants,
            is_top500
        ]]

        # Get model prediction
        prediction = model.predict(features)
        print("Model Prediction:", prediction)

        # ---- CUSTOM FAIL CONDITION FOR DEMO ----
        # If funding is too low OR no milestones → Force FAIL
        if funding_total_usd < 50000 or milestones == 0:
            result = "Startup will FAIL ❌"
        else:
            if prediction[0] == 1:
                result = "Startup will be SUCCESSFUL ✅"
            else:
                result = "Startup will FAIL ❌"

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return render_template("result.html", prediction_text="Error: " + str(e))


if __name__ == "__main__":
    app.run(debug=True)
