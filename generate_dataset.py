"""
Smart Attendance AI - Dataset Generator (Enhanced)
====================================================
Generates a synthetic but realistic student attendance dataset.

Enhancements over the original version:
- Command-line configurable (student count, seed, output path)
- Extra engineered signal columns (attendance consistency, at-risk subject count)
- Structured logging instead of bare prints
- Saves a JSON data dictionary describing every column
- Basic self-validation of the generated data before saving
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("generate_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic attendance data")
    parser.add_argument("--num-students", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="dataset/attendance_data.csv")
    return parser.parse_args()


def generate_dataset(num_students: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    student_ids = [f"STU{1001 + i}" for i in range(num_students)]
    names = [f"Student_{i + 1}" for i in range(num_students)]

    departments = rng.choice(
        ["Computer Science", "Data Analytics", "BCA", "Information Technology"],
        num_students,
    )
    semesters = rng.choice([3, 4, 5, 6], num_students)

    total_classes = rng.integers(60, 101, num_students)
    attendance_percentage = rng.uniform(55, 98, num_students)
    classes_attended = (total_classes * attendance_percentage / 100).round().astype(int)

    def jitter(base, spread):
        return np.clip(base + rng.normal(0, spread, num_students), 40, 100).round(2)

    math_attendance = jitter(attendance_percentage, 5)
    python_attendance = jitter(attendance_percentage, 5)
    dbms_attendance = jitter(attendance_percentage, 6)
    ai_attendance = jitter(attendance_percentage, 5)

    internal_marks = np.clip(
        attendance_percentage * 0.45 + rng.normal(45, 8, num_students), 30, 100
    ).round(2)
    assignment_percentage = np.clip(
        attendance_percentage * 0.55 + rng.normal(40, 10, num_students), 20, 100
    ).round(2)
    previous_attendance = np.clip(
        attendance_percentage + rng.normal(0, 8, num_students), 40, 100
    ).round(2)

    medical_leave = rng.integers(0, 11, num_students)

    subject_matrix = np.column_stack(
        [math_attendance, python_attendance, dbms_attendance, ai_attendance]
    )
    average_subject_attendance = subject_matrix.mean(axis=1).round(2)

    # --- New engineered signals -------------------------------------------------
    # How spread out a student's subject attendance is (volatility across subjects)
    attendance_consistency = subject_matrix.std(axis=1).round(2)

    # How many subjects the student is below a "risky" 65% threshold in
    at_risk_subject_count = (subject_matrix < 65).sum(axis=1)

    attendance_trend = np.where(
        attendance_percentage - previous_attendance > 3,
        "Improving",
        np.where(attendance_percentage - previous_attendance < -3, "Declining", "Stable"),
    )

    future_attendance_score = (
        attendance_percentage * 0.50
        + average_subject_attendance * 0.20
        + previous_attendance * 0.10
        + internal_marks * 0.05
        + assignment_percentage * 0.05
        + (100 - medical_leave * 2) * 0.10
    )
    future_attendance_score += rng.normal(0, 3, num_students)

    risk_category = np.where(
        future_attendance_score >= 80,
        "Low Risk",
        np.where(future_attendance_score >= 70, "Medium Risk", "High Risk"),
    )

    data = pd.DataFrame(
        {
            "Student_ID": student_ids,
            "Student_Name": names,
            "Department": departments,
            "Semester": semesters,
            "Total_Classes": total_classes,
            "Classes_Attended": classes_attended,
            "Attendance_Percentage": attendance_percentage.round(2),
            "Math_Attendance": math_attendance,
            "Python_Attendance": python_attendance,
            "DBMS_Attendance": dbms_attendance,
            "AI_Attendance": ai_attendance,
            "Internal_Marks": internal_marks,
            "Assignment_Percentage": assignment_percentage,
            "Previous_Attendance": previous_attendance,
            "Medical_Leave": medical_leave,
            "Average_Subject_Attendance": average_subject_attendance,
            "Attendance_Consistency": attendance_consistency,
            "At_Risk_Subject_Count": at_risk_subject_count,
            "Attendance_Trend": attendance_trend,
            "Risk_Category": risk_category,
        }
    )

    # Inject a few missing values so the cleaning script has real work to do
    data.loc[rng.choice(data.index, 10, replace=False), "Internal_Marks"] = np.nan
    data.loc[rng.choice(data.index, 8, replace=False), "Assignment_Percentage"] = np.nan

    return data


def validate(data: pd.DataFrame) -> None:
    """Cheap sanity checks so a broken generator fails loudly, not silently."""
    percentage_cols = [
        "Attendance_Percentage",
        "Math_Attendance",
        "Python_Attendance",
        "DBMS_Attendance",
        "AI_Attendance",
        "Previous_Attendance",
    ]
    for col in percentage_cols:
        out_of_range = data[col].dropna().between(0, 100).all()
        if not out_of_range:
            raise ValueError(f"Column '{col}' has values outside the 0-100 range")

    if data["Student_ID"].duplicated().any():
        raise ValueError("Duplicate Student_ID values were generated")

    log.info("Validation passed: ranges and uniqueness look correct.")


def write_data_dictionary(path: str) -> None:
    dictionary = {
        "Student_ID": "Unique student identifier",
        "Department": "Academic department",
        "Semester": "Current semester (3-6)",
        "Attendance_Percentage": "Overall attendance percentage",
        "Math_Attendance / Python_Attendance / DBMS_Attendance / AI_Attendance": "Subject-wise attendance percentage",
        "Internal_Marks": "Internal exam performance percentage",
        "Assignment_Percentage": "Assignment completion/performance percentage",
        "Previous_Attendance": "Attendance percentage in the prior term",
        "Medical_Leave": "Number of medical leave days taken",
        "Average_Subject_Attendance": "Mean of the four subject attendance columns",
        "Attendance_Consistency": "Std. deviation of subject attendance (higher = more erratic)",
        "At_Risk_Subject_Count": "Count of subjects below 65% attendance",
        "Attendance_Trend": "Improving / Stable / Declining vs. previous term",
        "Risk_Category": "Target label: Low Risk / Medium Risk / High Risk",
    }
    with open(path, "w") as f:
        json.dump(dictionary, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    log.info("Generating dataset for %d students (seed=%d)...", args.num_students, args.seed)
    data = generate_dataset(args.num_students, args.seed)
    validate(data)

    data.to_csv(args.output, index=False)
    dict_path = os.path.join(os.path.dirname(args.output), "data_dictionary.json")
    write_data_dictionary(dict_path)

    log.info("Dataset saved to %s", args.output)
    log.info("Data dictionary saved to %s", dict_path)
    log.info("Records: %d | Columns: %d", *data.shape)
    log.info("Risk Category Distribution:\n%s", data["Risk_Category"].value_counts().to_string())


if __name__ == "__main__":
    main()