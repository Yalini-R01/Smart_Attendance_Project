"""
Smart Attendance AI - Flask App (Enhanced)
============================================
Enhancements over the original version:
- Server-side input validation (range checks) with user-facing error messages
  instead of raw 500 errors on bad input
- Structured logging of every prediction
- SQLite-backed prediction history (persists across restarts), with a
  /history page to browse it and a way to clear it
- Confidence score (predict_proba) shown alongside the predicted risk
- A JSON /health endpoint and a friendly 404/500 error page
"""

import logging
import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, flash, g, jsonify, redirect, render_template, request, url_for

# =========================================
# PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

# =========================================
# LOGGING
# =========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("smart_attendance_app")

# =========================================
# FLASK APP
# =========================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

# =========================================
# LOAD MODEL ARTIFACTS
# =========================================

MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "features.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    log.info("Model, scaler, and feature list loaded successfully.")
except FileNotFoundError:
    model = scaler = features = None
    log.warning(
        "Model artifacts not found at %s. Run train_model.py first - "
        "the app will start but /predict will show a friendly error.",
        os.path.join(BASE_DIR, "model"),
    )

# =========================================
# FIELD DEFINITIONS (label + valid range), used for both validation and
# rendering error messages consistently
# =========================================

FIELDS = {
    "attendance": ("Overall Attendance", 0, 100),
    "math": ("Mathematics Attendance", 0, 100),
    "python": ("Python Attendance", 0, 100),
    "dbms": ("DBMS Attendance", 0, 100),
    "ai": ("AI Attendance", 0, 100),
    "internal": ("Internal Marks", 0, 100),
    "assignment": ("Assignment Performance", 0, 100),
    "previous": ("Previous Attendance", 0, 100),
    "medical_leave": ("Medical Leave (days)", 0, 365),
}


# =========================================
# DATABASE HELPERS
# =========================================

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            attendance REAL, math REAL, python REAL, dbms REAL, ai REAL,
            internal REAL, assignment REAL, previous REAL, medical_leave REAL,
            prediction TEXT NOT NULL,
            confidence REAL
        )
        """
    )
    conn.commit()
    conn.close()


# =========================================
# VALIDATION
# =========================================

def validate_inputs(form) -> tuple[dict, list]:
    """Parses and range-checks the form. Returns (values, error_messages)."""
    values = {}
    errors = []

    for field, (label, min_val, max_val) in FIELDS.items():
        raw = form.get(field, "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            continue
        try:
            number = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not (min_val <= number <= max_val):
            errors.append(f"{label} must be between {min_val} and {max_val}.")
            continue
        values[field] = number

    return values, errors


# =========================================
# ROUTES
# =========================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("The model isn't trained yet. Run train_model.py, then restart the app.", "error")
        return redirect(url_for("home"))

    values, errors = validate_inputs(request.form)
    if errors:
        for message in errors:
            flash(message, "error")
        return redirect(url_for("home"))

    attendance = values["attendance"]
    math = values["math"]
    python = values["python"]
    dbms = values["dbms"]
    ai = values["ai"]
    internal = values["internal"]
    assignment = values["assignment"]
    previous = values["previous"]
    medical_leave = values["medical_leave"]

    overall_attendance = (math + python + dbms + ai) / 4
    attendance_change = attendance - previous
    academic_performance = (internal + assignment) / 2

    input_data = pd.DataFrame(
        [
            {
                "Attendance_Percentage": attendance,
                "Math_Attendance": math,
                "Python_Attendance": python,
                "DBMS_Attendance": dbms,
                "AI_Attendance": ai,
                "Internal_Marks": internal,
                "Assignment_Percentage": assignment,
                "Previous_Attendance": previous,
                "Medical_Leave": medical_leave,
                "Overall_Attendance": overall_attendance,
                "Attendance_Change": attendance_change,
                "Academic_Performance": academic_performance,
            }
        ]
    )
    input_data = input_data[features]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = round(float(max(probabilities)) * 100, 1)

    log.info(
        "Prediction=%s confidence=%s attendance=%.1f overall_subject_avg=%.1f",
        prediction, confidence, attendance, overall_attendance,
    )

    db = get_db()
    db.execute(
        """
        INSERT INTO predictions
            (created_at, attendance, math, python, dbms, ai, internal, assignment,
             previous, medical_leave, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            attendance, math, python, dbms, ai, internal, assignment,
            previous, medical_leave, prediction, confidence,
        ),
    )
    db.commit()

    return render_template(
        "result.html",
        prediction=prediction,
        attendance=attendance,
        confidence=confidence,
    )


@app.route("/history")
def history():
    db = get_db()
    risk_filter = request.args.get("risk")

    query = "SELECT * FROM predictions"
    params = ()
    if risk_filter in ("Low Risk", "Medium Risk", "High Risk"):
        query += " WHERE prediction = ?"
        params = (risk_filter,)
    query += " ORDER BY id DESC LIMIT 200"

    rows = db.execute(query, params).fetchall()
    return render_template("history.html", records=rows, active_filter=risk_filter)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    db = get_db()
    db.execute("DELETE FROM predictions")
    db.commit()
    flash("Prediction history cleared.", "success")
    return redirect(url_for("history"))


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok" if model is not None else "model_missing",
            "model_loaded": model is not None,
        }
    )


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("error.html", code=500, message="Something went wrong."), 500


# =========================================
# ENTRYPOINT
# =========================================

if __name__ == "__main__":
    init_db()
    app.run(debug=True)