# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 2. LOAD DATA
# =========================
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

# =========================
# 3. CLEANING
# =========================
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.fillna(df.median(), inplace=True)

# Convert target
df.iloc[:, -1] = (df.iloc[:, -1] > 0).astype(int)

# =========================
# 4. EDA VISUALS
# =========================
plt.figure()
sns.countplot(x=df.iloc[:, -1])
plt.title("Heart Disease Distribution")
plt.savefig("target_distribution.png")

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation.png")

# =========================
# 5. TRAIN MODEL
# =========================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train_scaled, y_train)

# =========================
# 6. EVALUATION
# =========================
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# 7. FEATURE IMPORTANCE
# =========================
importance = model.feature_importances_
feat_names = X.columns

feat_df = pd.DataFrame({
    "Feature": feat_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")

# =========================
# 8. SAVE EVERYTHING
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
df.to_csv("cleaned_data.csv", index=False)

print("✅ Model, scaler, and dataset saved!")