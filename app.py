from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
priority_model = joblib.load("priority.joblib")
category_model = joblib.load("category.joblib")

@app.route("/", methods=["GET", "POST"])
def home():

    # GET request → just show upload page
    if request.method == "GET":
        return render_template("index.html")

    # POST request → CSV uploaded
    file = request.files["csv_file"]

    # Read uploaded CSV
    df_raw = pd.read_csv(file, header=None)

    df = df_raw[0].str.replace('"', '').str.split(",", expand=True)

    df.columns = [
        "ticket_id",
        "emp_id",
        "department",
        "business_unit",
        "ticket_date",
        "ticket_text",
        "category",
        "priority"
    ]

    # Predictions
    df["predicted_priority"] = priority_model.predict(df["ticket_text"])
    df["predicted_category"] = category_model.predict(df["ticket_text"])

    priority_order = {"High": 1, "Moderate": 2, "Low": 3}
    df["priority_rank"] = df["predicted_priority"].map(priority_order)

    df = df.sort_values("priority_rank")

    results = []
    for _, row in df.iterrows():
        results.append({
            "emp_id": row["emp_id"],
            "department": row["department"],
            "ticket_date": row["ticket_date"],
            "ticket_text": row["ticket_text"],
            "priority": row["predicted_priority"],
            "category": row["predicted_category"]
        })

    return render_template("results.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
