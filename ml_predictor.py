import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Εκκίνηση εκπαίδευσης...")

# Φόρτωση dataset
try:
    df = pd.read_csv("topcoder_features_extracted_LOCAL.csv")
except FileNotFoundError:
    print("Σφάλμα: Δεν βρέθηκε το αρχείο csv.")
    exit()

TARGET_COLUMN = 'Prize_USD' 

# Καθαρισμός nulls
df = df.dropna(subset=[TARGET_COLUMN])
df = df.fillna(0)

# Αφαίρεση outliers για να μην χαλάσει το μοντέλο (κρατάμε 50-2500$)
df = df[(df[TARGET_COLUMN] >= 50) & (df[TARGET_COLUMN] <= 2500)]
print(f"Έργα μετά το φιλτράρισμα: {len(df)}")

# Features & Target (χρησιμοποιούμε μόνο llama λόγω hardware constraints)
X = df[['Complexity_Llama', 'Tech_Count_Llama', 'QA_Effort_Llama', 'Deliverables_Count_Llama']]
y = df[TARGET_COLUMN]

# Train/Test split (80-20)
if len(df) > 5:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = X, X, y, y

# Εκπαίδευση Linear Regression (Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Εκπαίδευση Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Υπολογισμός MAE & R2
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nΑποτελέσματα Αξιολόγησης:")
print("-" * 40)
print("Linear Regression")
print(f" - MAE: ${lr_mae:.2f}")
print(f" - R² Score: {lr_r2:.2f}")
print("\nRandom Forest")
print(f" - MAE: ${rf_mae:.2f}")
print(f" - R² Score: {rf_r2:.2f}")
print("-" * 40)

print("\nΕξαγωγή γραφημάτων...")
sns.set_theme(style="whitegrid")

# Plot 1: Feature Importance
plt.figure(figsize=(8, 5))
importances = rf_model.feature_importances_ * 100
clean_features = ['Complexity', 'Tech Count', 'QA Effort', 'Deliverables Count']
sns.barplot(x=importances, y=clean_features, hue=clean_features, palette="viridis", legend=False)

plt.title("Σημαντικότητα Χαρακτηριστικών στο Κόστος (Random Forest)", fontsize=14, fontweight='bold')
plt.xlabel("Συμμετοχή στη διαμόρφωση της τιμής (%)", fontsize=12)
plt.ylabel("Μετρικές Llama 3.2", fontsize=12)
plt.tight_layout()
plt.savefig("chart_1_feature_importance.png", dpi=300)
plt.close()

# Plot 2: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_predictions, alpha=0.6, color='darkorange', edgecolor='k', s=60, label='Linear Regression')
plt.scatter(y_test, rf_predictions, alpha=0.7, color='royalblue', edgecolor='k', s=60, label='Random Forest')

max_val = max(y_test.max(), rf_predictions.max(), lr_predictions.max()) if len(y_test) > 0 else 100
if max_val == 0 or pd.isna(max_val): max_val = 100
plt.plot([0, max_val], [0, max_val], '--r', linewidth=2, label='Τέλεια Πρόβλεψη (Σφάλμα 0)')

plt.title("Σύγκριση Μοντέλων: Πραγματικό vs Προβλεπόμενο Κόστος", fontsize=14, fontweight='bold')
plt.xlabel("Πραγματική Αμοιβή (USD)", fontsize=12)
plt.ylabel("Πρόβλεψη Αλγορίθμου (USD)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("chart_2_model_comparison.png", dpi=300)
plt.close()

print("Ολοκληρώθηκε.")