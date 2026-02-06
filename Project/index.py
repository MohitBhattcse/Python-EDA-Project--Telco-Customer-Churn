import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import math

# Load data
Data = pd.read_csv("C:/Users/Mohit.Bhatt/OneDrive - GEP/Desktop/Python/Project/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ---- Data Cleaning / Preparation ----
# Replace blank TotalCharges with 0 and convert to float if needed
# Data["TotalCharges"] = Data["TotalCharges"].replace(" ", 0)
# Data["TotalCharges"] = Data["TotalCharges"].astype("float")

# Check for nulls
# print(Data.isnull().sum().sum()) # 0 null values

# Check for duplicates
# print(Data.duplicated().sum())  # Zero duplicates
# print(Data["customerID"].duplicated().sum()) # Zero duplicates

# Convert SeniorCitizen from 0/1 to No/Yes
# def conv(value):
#     if value == 1:
#         return "Yes"
#     else:
#         return "No"

# Data['SeniorCitizen'] = Data["SeniorCitizen"].apply(conv)

# ---- Basic Churn Analysis ----
# Count of Churn
plt.figure(figsize=(4,4))
ax = sn.countplot(data=Data, x="Churn")
ax.bar_label(ax.containers[0])
plt.title("Count of Customers by Churn")
plt.show()

# Churn percentage
# Insight: ~26.5% customers churned, majority stayed
group=Data.groupby("SeniorCitizen")["SeniorCitizen"].count()
plt.pie(group.values,labels=group.index,autopct="%1.1f%%")
plt.title("Total no of senior citizens")
plt.show()
# ---- Churn by Gender ----
plt.figure(figsize=(3,3))
ax = sn.countplot(data=Data, x="gender", hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Gender")
plt.show()
# Insight: Gender does not significantly affect churn

# ---- Churn by Senior Citizen ----
plt.figure(figsize=(3,3))
ax = sn.countplot(data=Data, x="SeniorCitizen", hue="Churn")
for container in ax.containers:
    ax.bar_label(container, label_type="edge")
plt.title("Churn by Senior Citizen")
plt.show()

# Senior citizen churn in percentage
total = len(Data)
plt.figure(figsize=(3,3))
ax = sn.countplot(data=Data, x="SeniorCitizen", hue="Churn")
for container in ax.containers:
    labels = [f"{(bar.get_height()/total)*100:.1f}%" for bar in container]
    ax.bar_label(container, labels=labels)
plt.title("Churn by Senior Citizen (%)")
plt.tight_layout()
plt.show()

# Row-normalized percentages by SeniorCitizen
tbl = pd.crosstab(Data["SeniorCitizen"], Data["Churn"], normalize="index") * 100
tbl = tbl.reset_index().melt(id_vars="SeniorCitizen", var_name="Churn", value_name="pct")
plt.figure(figsize=(3,3))
ax = sn.barplot(data=tbl, x="SeniorCitizen", y="pct", hue="Churn")
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f%%", padding=2)
plt.ylabel("Percentage")
plt.title("Churn by Senior Citizen (within-group %)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# Pie chart of total Senior Citizens
group = Data.groupby("SeniorCitizen")["SeniorCitizen"].count()
plt.pie(group.values, labels=group.index, autopct="%1.1f%%")
plt.title("Total Number of Senior Citizens")
plt.show()

# ---- Tenure Analysis ----
sn.histplot(data=Data, x="tenure", bins=72, hue="Churn")
plt.title("Customer Tenure Distribution")
plt.show()
# Insight: Short-tenure customers churn more; long-tenure customers stay

# ---- Contract Analysis ----
a = sn.countplot(data=Data, x="Contract", hue="Churn")
for container in a.containers:
    plt.bar_label(container, label_type="edge")
plt.title("Churn by Contract Type")
plt.show()
# Insight: Month-to-month contracts have high churn; 1 or 2-year contracts have low churn

# ---- Service Usage Analysis ----
columns = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

n_plots = len(columns)
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)

plt.figure(figsize=(18, 12))
for i, column in enumerate(columns):
    plt.subplot(n_rows, n_cols, i + 1)
    ax = sn.countplot(data=Data, x=column, hue="Churn")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
plt.tight_layout()
plt.show()

"""
Service Insights:
- PhoneService / MultipleLines → No significant impact on churn
- InternetService → Fiber optic users churn more
- OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport → reduce churn
- StreamingTV / StreamingMovies → minimal effect on churn
"""

# ---- Billing & Payment Analysis ----
columns1 = ["PaperlessBilling", "PaymentMethod"]
n_len = len(columns1)
n_col = 2
no_rows = math.ceil(n_len / n_col)

plt.figure(figsize=(11,12))
for i, column in enumerate(columns1):
    plt.subplot(no_rows, n_col, i+1)
    ax = sn.countplot(data=Data, x=column, hue="Churn")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""
Billing Insights:
- PaperlessBilling → higher churn
- PaymentMethod → Automatic payments reduce churn; Electronic/Mailed checks increase churn
"""

# ---- Monthly Charges & Total Charges ----
columns1 = ["MonthlyCharges", "TotalCharges"]
plt.figure(figsize=(11,12))
for i, column in enumerate(columns1):
    plt.subplot(no_rows, n_col, i+1)
    ax = sn.histplot(data=Data, x=column, hue="Churn", kde=True)
    # Histogram label optional
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""
Charges Insights:
1. MonthlyCharges:
   - Low ($20-30) → mostly non-churners
   - Mid-High ($70-100) → higher churn
   - Insight: High-paying customers are more likely to churn (price sensitivity)
2. TotalCharges:
   - Very low (0-500) → high churn (early leavers)
   - High (>3000-4000) → low churn (loyal long-tenure customers)
3. Combined Insight:
   - High MonthlyCharges + Low TotalCharges → high-risk churners
   - These customers are expensive AND new → most vulnerable
"""

# ---- End of EDA ----

