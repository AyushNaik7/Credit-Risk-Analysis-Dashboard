# ==========================================
# CREDIT RISK ANALYSIS PROJECT
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Credit Risk Analysis Project Started")


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("credit_risk_dataset.csv")


# ==========================================
# 2. BASIC DATA EXPLORATION
# ==========================================

print("\nFirst 5 Rows:")
print(df.head())


print("\nShape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nDataset Information:")
print(df.info())


print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 4. DUPLICATE ROWS
# ==========================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())
# ============================================================
# CREDIT RISK ANALYSIS
# Python + Pandas + Matplotlib + Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("CREDIT RISK ANALYSIS PROJECT")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("credit_risk_dataset.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 3. CHECK DUPLICATES
# ============================================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing employment length with median
if "person_emp_length" in df.columns:
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )

# Fill missing interest rate with median
if "loan_int_rate" in df.columns:
    df["loan_int_rate"] = df["loan_int_rate"].fillna(
        df["loan_int_rate"].median()
    )

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 5. BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

if "person_age" in df.columns:
    print("\nAge Range:")
    print(df["person_age"].min(), "to", df["person_age"].max())

if "loan_amnt" in df.columns:
    print("\nLoan Amount Range:")
    print(df["loan_amnt"].min(), "to", df["loan_amnt"].max())

if "loan_status" in df.columns:
    print("\nLoan Status Values:")
    print(df["loan_status"].value_counts())


# ============================================================
# 6. CREATE RISK CATEGORY
# ============================================================

# Create a simple risk category using loan_percent_income
# This is an analytical segmentation, NOT a medical/financial
# prediction model.

if "loan_percent_income" in df.columns:

    def risk_category(x):
        if x >= 0.40:
            return "High Risk"
        elif x >= 0.20:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_category"] = df["loan_percent_income"].apply(risk_category)

print("\n" + "=" * 60)
print("RISK CATEGORY")
print("=" * 60)

print(df["risk_category"].value_counts())


# ============================================================
# 7. OVERALL LOAN DEFAULT ANALYSIS
# ============================================================

if "loan_status" in df.columns:

    default_count = df["loan_status"].value_counts()

    print("\nLoan Status:")
    print(default_count)

    default_percentage = (
        df["loan_status"].value_counts(normalize=True) * 100
    ).round(2)

    print("\nLoan Status Percentage:")
    print(default_percentage)


# ============================================================
# 8. DEFAULT RATE
# ============================================================

if "loan_status" in df.columns:

    overall_default_rate = df["loan_status"].mean() * 100

    print("\nOverall Default Rate:")
    print(round(overall_default_rate, 2), "%")


# ============================================================
# 9. DEFAULT RATE BY LOAN GRADE
# ============================================================

if "loan_grade" in df.columns and "loan_status" in df.columns:

    grade_default = (
        df.groupby("loan_grade")["loan_status"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY LOAN GRADE (%)")
    print("=" * 60)

    print(grade_default)


# ============================================================
# 10. DEFAULT RATE BY LOAN INTENT
# ============================================================

if "loan_intent" in df.columns and "loan_status" in df.columns:

    intent_default = (
        df.groupby("loan_intent")["loan_status"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY LOAN INTENT (%)")
    print("=" * 60)

    print(intent_default)


# ============================================================
# 11. DEFAULT RATE BY HOME OWNERSHIP
# ============================================================

if "person_home_ownership" in df.columns and "loan_status" in df.columns:

    home_default = (
        df.groupby("person_home_ownership")["loan_status"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY HOME OWNERSHIP (%)")
    print("=" * 60)

    print(home_default)


# ============================================================
# 12. DEFAULT RATE BY PREVIOUS DEFAULT HISTORY
# ============================================================

if (
    "cb_person_default_on_file" in df.columns
    and "loan_status" in df.columns
):

    previous_default = (
        df.groupby("cb_person_default_on_file")["loan_status"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY PREVIOUS DEFAULT HISTORY (%)")
    print("=" * 60)

    print(previous_default)


# ============================================================
# 13. DEFAULT RATE BY RISK CATEGORY
# ============================================================

if "risk_category" in df.columns and "loan_status" in df.columns:

    risk_default = (
        df.groupby("risk_category")["loan_status"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY RISK CATEGORY (%)")
    print("=" * 60)

    print(risk_default)


# ============================================================
# 14. AVERAGE LOAN AMOUNT BY DEFAULT STATUS
# ============================================================

if "loan_amnt" in df.columns and "loan_status" in df.columns:

    avg_loan = (
        df.groupby("loan_status")["loan_amnt"]
        .mean()
        .round(2)
    )

    print("\n" + "=" * 60)
    print("AVERAGE LOAN AMOUNT BY STATUS")
    print("=" * 60)

    print(avg_loan)


# ============================================================
# 15. AVERAGE INCOME BY DEFAULT STATUS
# ============================================================

if "person_income" in df.columns and "loan_status" in df.columns:

    avg_income = (
        df.groupby("loan_status")["person_income"]
        .mean()
        .round(2)
    )

    print("\n" + "=" * 60)
    print("AVERAGE INCOME BY DEFAULT STATUS")
    print("=" * 60)

    print(avg_income)


# ============================================================
# 16. AVERAGE INTEREST RATE BY DEFAULT STATUS
# ============================================================

if "loan_int_rate" in df.columns and "loan_status" in df.columns:

    avg_interest = (
        df.groupby("loan_status")["loan_int_rate"]
        .mean()
        .round(2)
    )

    print("\n" + "=" * 60)
    print("AVERAGE INTEREST RATE BY DEFAULT STATUS")
    print("=" * 60)

    print(avg_interest)


# ============================================================
# 17. AVERAGE LOAN-TO-INCOME PERCENTAGE BY DEFAULT STATUS
# ============================================================

if (
    "loan_percent_income" in df.columns
    and "loan_status" in df.columns
):

    avg_loan_income = (
        df.groupby("loan_status")["loan_percent_income"]
        .mean()
        .round(3)
    )

    print("\n" + "=" * 60)
    print("AVERAGE LOAN % OF INCOME BY STATUS")
    print("=" * 60)

    print(avg_loan_income)


# ============================================================
# 18. AGE GROUP ANALYSIS
# ============================================================

if "person_age" in df.columns:

    df["age_group"] = pd.cut(
        df["person_age"],
        bins=[0, 25, 35, 50, 100],
        labels=[
            "18-25",
            "26-35",
            "36-50",
            "51+"
        ]
    )

    print("\n" + "=" * 60)
    print("CUSTOMER COUNT BY AGE GROUP")
    print("=" * 60)

    print(df["age_group"].value_counts().sort_index())


# ============================================================
# 19. DEFAULT RATE BY AGE GROUP
# ============================================================

if (
    "age_group" in df.columns
    and "loan_status" in df.columns
):

    age_default = (
        df.groupby("age_group", observed=True)["loan_status"]
        .mean()
        .mul(100)
        .round(2)
    )

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY AGE GROUP (%)")
    print("=" * 60)

    print(age_default)


# ============================================================
# 20. CORRELATION ANALYSIS
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation = df[numeric_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION WITH LOAN STATUS")
print("=" * 60)

if "loan_status" in correlation.columns:

    print(
        correlation["loan_status"]
        .sort_values(ascending=False)
    )


# ============================================================
# 21. VISUALIZATION SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")


# ============================================================
# CHART 1 — LOAN DEFAULT DISTRIBUTION
# ============================================================

if "loan_status" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="loan_status"
    )

    plt.title("Loan Default Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Number of Loans")

    plt.tight_layout()

    plt.savefig(
        "01_loan_default_distribution.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 2 — DEFAULT RATE BY LOAN GRADE
# ============================================================

if "loan_grade" in df.columns and "loan_status" in df.columns:

    plt.figure(figsize=(9, 5))

    grade_default.sort_values().plot(
        kind="bar"
    )

    plt.title("Default Rate by Loan Grade")
    plt.xlabel("Loan Grade")
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "02_default_rate_by_loan_grade.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 3 — DEFAULT RATE BY LOAN INTENT
# ============================================================

if "loan_intent" in df.columns and "loan_status" in df.columns:

    plt.figure(figsize=(10, 5))

    intent_default.sort_values().plot(
        kind="bar"
    )

    plt.title("Default Rate by Loan Intent")
    plt.xlabel("Loan Intent")
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        "03_default_rate_by_loan_intent.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 4 — DEFAULT RATE BY HOME OWNERSHIP
# ============================================================

if (
    "person_home_ownership" in df.columns
    and "loan_status" in df.columns
):

    plt.figure(figsize=(8, 5))

    home_default.sort_values().plot(
        kind="bar"
    )

    plt.title("Default Rate by Home Ownership")
    plt.xlabel("Home Ownership")
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "04_default_rate_by_home_ownership.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 5 — DEFAULT RATE BY RISK CATEGORY
# ============================================================

if "risk_category" in df.columns and "loan_status" in df.columns:

    risk_order = [
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]

    risk_plot = (
        df.groupby("risk_category", observed=True)["loan_status"]
        .mean()
        .mul(100)
        .reindex(risk_order)
    )

    plt.figure(figsize=(8, 5))

    risk_plot.plot(
        kind="bar"
    )

    plt.title("Default Rate by Risk Category")
    plt.xlabel("Risk Category")
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "05_default_rate_by_risk_category.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 6 — LOAN AMOUNT VS INCOME
# ============================================================

if (
    "person_income" in df.columns
    and "loan_amnt" in df.columns
    and "loan_status" in df.columns
):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df.sample(
            min(5000, len(df)),
            random_state=42
        ),
        x="person_income",
        y="loan_amnt",
        hue="loan_status",
        alpha=0.6
    )

    plt.title("Loan Amount vs Income")
    plt.xlabel("Annual Income")
    plt.ylabel("Loan Amount")

    plt.tight_layout()

    plt.savefig(
        "06_loan_amount_vs_income.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 7 — INTEREST RATE VS LOAN STATUS
# ============================================================

if (
    "loan_int_rate" in df.columns
    and "loan_status" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="loan_status",
        y="loan_int_rate"
    )

    plt.title("Interest Rate Distribution by Loan Status")
    plt.xlabel("Loan Status")
    plt.ylabel("Interest Rate (%)")

    plt.tight_layout()

    plt.savefig(
        "07_interest_rate_by_status.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 8 — LOAN % INCOME VS DEFAULT
# ============================================================

if (
    "loan_percent_income" in df.columns
    and "loan_status" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="loan_status",
        y="loan_percent_income"
    )

    plt.title("Loan-to-Income Ratio by Loan Status")
    plt.xlabel("Loan Status")
    plt.ylabel("Loan % of Income")

    plt.tight_layout()

    plt.savefig(
        "08_loan_percent_income_by_status.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 9 — DEFAULT RATE BY PREVIOUS DEFAULT HISTORY
# ============================================================

if (
    "cb_person_default_on_file" in df.columns
    and "loan_status" in df.columns
):

    plt.figure(figsize=(8, 5))

    previous_default.sort_values().plot(
        kind="bar"
    )

    plt.title("Default Rate by Previous Default History")
    plt.xlabel("Previous Default on File")
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "09_default_rate_previous_default.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CHART 10 — CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "10_correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. SAVE CLEANED DATA
# ============================================================

df.to_csv(
    "cleaned_credit_risk_dataset.csv",
    index=False
)


# ============================================================
# 23. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

if "loan_status" in df.columns:
    print(
        "Overall Default Rate:",
        round(df["loan_status"].mean() * 100, 2),
        "%"
    )

if "risk_category" in df.columns:
    print("\nRisk Category Distribution:")
    print(df["risk_category"].value_counts())

print("\nCleaned dataset saved as:")
print("cleaned_credit_risk_dataset.csv")

print("\nCharts saved successfully.")

print("\n" + "=" * 60)
print("CREDIT RISK ANALYSIS COMPLETED")
print("=" * 60)