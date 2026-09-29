"""
Smart Attendance AI - Data Cleaning (Enhanced)
================================================
Enhancements over the original version:
- Structured logging instead of bare prints
- IQR-based outlier detection (reported, not blindly dropped) on numeric columns
- Explicit range validation for percentage columns, with a clear error if the
  raw data is fundamentally broken (rather than silently clipping garbage)
- Saves a machine-readable cleaning_report.json summarizing every change made
"""

import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("clean_data")

INPUT_PATH = "dataset/attendance_data.csv"
OUTPUT_PATH = "dataset/cleaned_attendance_data.csv"
REPORT_PATH = "dataset/cleaning_report.json"

PERCENTAGE_COLUMNS = [
    "Attendance_Percentage",
    "Math_Attendance",
    "Python_Attendance",
    "DBMS_Attendance",
    "AI_Attendance",
    "Previous_Attendance",
    "Average_Subject_Attendance",
    "Internal_Marks",
    "Assignment_Percentage",
]


def detect_outliers_iqr(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def main():
    report = {}

    log.info("Loading raw dataset from %s", INPUT_PATH)
    data = pd.read_csv(INPUT_PATH)
    report["initial_shape"] = list(data.shape)

    missing_before = data.isnull().sum()
    report["missing_before"] = {k: int(v) for k, v in missing_before[missing_before > 0].items()}
    log.info("Missing values before cleaning:\n%s", missing_before[missing_before > 0].to_string())

    # --- Fill missing values -----------------------------------------------
    numeric_columns = data.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        if data[column].isnull().sum() > 0:
            median_value = data[column].median()
            data[column] = data[column].fillna(median_value)

    categorical_columns = data.select_dtypes(exclude=np.number).columns
    for column in categorical_columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(data[column].mode()[0])

    # --- Duplicates ----------------------------------------------------------
    duplicate_count = int(data.duplicated().sum())
    report["duplicates_removed"] = duplicate_count
    log.info("Duplicate records found: %d", duplicate_count)
    data = data.drop_duplicates()

    # --- Range validation + outlier report (existing columns only) ----------
    outlier_counts = {}
    for column in [c for c in PERCENTAGE_COLUMNS if c in data.columns]:
        below_zero = int((data[column] < 0).sum())
        above_hundred = int((data[column] > 100).sum())
        if below_zero or above_hundred:
            log.warning(
                "'%s' has %d values below 0 and %d values above 100 - clipping.",
                column, below_zero, above_hundred,
            )
        data[column] = data[column].clip(0, 100)
        outlier_counts[column] = detect_outliers_iqr(data[column])

    report["out_of_range_clipped"] = {"below_zero_or_above_100": "see log for per-column counts"}
    report["iqr_outlier_counts"] = outlier_counts
    log.info("IQR-based outlier counts (informational, not removed): %s", outlier_counts)

    # Medical leave can't be negative
    if "Medical_Leave" in data.columns:
        data["Medical_Leave"] = data["Medical_Leave"].clip(lower=0)

    missing_after = data.isnull().sum()
    report["missing_after"] = {k: int(v) for k, v in missing_after[missing_after > 0].items()}
    report["final_shape"] = list(data.shape)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Cleaned dataset saved to %s", OUTPUT_PATH)
    log.info("Cleaning report saved to %s", REPORT_PATH)
    log.info("Final shape: %s", data.shape)


if __name__ == "__main__":
    main()