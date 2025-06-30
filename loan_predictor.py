"""
Loan Eligibility Predictor - Machine Learning Project
Created by Sujal Sariya as part of RISE Internship (Tamizhan Skills)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("🔁 Loading dataset...")

# Load the dataset
df = pd.read_csv("loan_data.csv")

print("✅ Sample Data:")
print(df.head())

# Preprocessing
print("🧹 Preprocessing data...")
df.fillna(method='ffill', inplace=True)
le = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

# Split features and labels
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
print("🧠 Training Random Forest Classifier...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📈 Classification Report:\n", classification_report(y_test, y_pred))
