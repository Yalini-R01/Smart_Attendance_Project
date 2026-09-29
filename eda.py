"""
Smart Attendance AI - Exploratory Data Analysis (Enhanced)
============================================================
Enhancements over the original version:
- Correlation heatmap across numeric features
- Attendance-consistency and at-risk-subject-count plots (new features)
- Plots saved without blocking on plt.show() (safe for headless/CI runs)
- A plain-text eda_summary.txt with the key numeric takeaways
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; still saves all PNGs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_PATH = "dataset/cleaned_attendance_data.csv"
GRAPH_DIR = "dataset/eda_graphs"
SUMMARY_PATH = "dataset/eda_summary.txt"


def save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def main():
    data = pd.read_csv(INPUT_PATH)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    print("Cleaned dataset loaded successfully!")
    print("\nDataset Information:")
    print(data.info())
    print("\nStatistical Summary:")
    print(data.describe())

    data.describe().to_csv(os.path.join(GRAPH_DIR, "summary_statistics.csv"))

    # 1. Attendance distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data["Attendance_Percentage"], bins=20, kde=True)
    plt.title("Distribution of Student Attendance")
    plt.xlabel("Attendance Percentage")
    plt.ylabel("Number of Students")
    save(os.path.join(GRAPH_DIR, "attendance_distribution.png"))

    # 2. Risk category distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="Risk_Category", order=["Low Risk", "Medium Risk", "High Risk"])
    plt.title("Student Risk Category Distribution")
    plt.xlabel("Risk Category")
    plt.ylabel("Number of Students")
    save(os.path.join(GRAPH_DIR, "risk_distribution.png"))

    # 3. Department-wise attendance
    plt.figure(figsize=(9, 5))
    sns.barplot(data=data, x="Department", y="Attendance_Percentage")
    plt.title("Department-wise Average Attendance")
    plt.xlabel("Department")
    plt.ylabel("Average Attendance (%)")
    plt.xticks(rotation=20)
    save(os.path.join(GRAPH_DIR, "department_attendance.png"))

    # 4. Subject-wise attendance
    subject_columns = ["Math_Attendance", "Python_Attendance", "DBMS_Attendance", "AI_Attendance"]
    subject_means = data[subject_columns].mean()
    plt.figure(figsize=(8, 5))
    subject_means.plot(kind="bar")
    plt.title("Average Subject-wise Attendance")
    plt.xlabel("Subject")
    plt.ylabel("Average Attendance (%)")
    plt.xticks(rotation=20)
    save(os.path.join(GRAPH_DIR, "subject_attendance.png"))

    # 5. Attendance vs internal marks
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=data, x="Attendance_Percentage", y="Internal_Marks", hue="Risk_Category",
        hue_order=["Low Risk", "Medium Risk", "High Risk"],
    )
    plt.title("Attendance vs Internal Marks")
    plt.xlabel("Attendance Percentage")
    plt.ylabel("Internal Marks")
    save(os.path.join(GRAPH_DIR, "attendance_vs_marks.png"))

    # 6. Attendance trend
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="Attendance_Trend", order=["Improving", "Stable", "Declining"])
    plt.title("Attendance Trend Distribution")
    plt.xlabel("Attendance Trend")
    plt.ylabel("Number of Students")
    save(os.path.join(GRAPH_DIR, "attendance_trend.png"))

    # 7. NEW: Correlation heatmap
    numeric_cols = data.select_dtypes(include="number").columns
    plt.figure(figsize=(11, 9))
    sns.heatmap(data[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    save(os.path.join(GRAPH_DIR, "correlation_heatmap.png"))

    # 8. NEW: At-risk subject count vs risk category (if present)
    if "At_Risk_Subject_Count" in data.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(
            data=data, x="Risk_Category", y="At_Risk_Subject_Count",
            order=["Low Risk", "Medium Risk", "High Risk"],
        )
        plt.title("Subjects Below 65% Attendance, by Risk Category")
        plt.xlabel("Risk Category")
        plt.ylabel("At-Risk Subject Count")
        save(os.path.join(GRAPH_DIR, "at_risk_subject_count.png"))

    # --- Written summary -----------------------------------------------------
    risk_counts = data["Risk_Category"].value_counts()
    top_corr = (
        data[numeric_cols].corr()["Attendance_Percentage"].drop("Attendance_Percentage").abs().sort_values(ascending=False)
    )

    lines = [
        "SMART ATTENDANCE AI - EDA SUMMARY",
        "=================================",
        f"Total students analyzed: {len(data)}",
        "",
        "Risk category distribution:",
        risk_counts.to_string(),
        "",
        "Features most correlated with Attendance_Percentage (abs value):",
        top_corr.head(5).to_string(),
        "",
        f"Overall mean attendance: {data['Attendance_Percentage'].mean():.2f}%",
        f"Overall mean internal marks: {data['Internal_Marks'].mean():.2f}%",
    ]
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))

    print("\n-----------------------------------------")
    print("EDA completed successfully!")
    print("-----------------------------------------")
    print(f"Graphs saved inside: {GRAPH_DIR}/")
    print(f"Text summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()