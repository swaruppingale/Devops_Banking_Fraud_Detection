import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🔄 Loading dataset...")
df = pd.read_csv("transactions.csv")

X = df[['amount','frequency','location_score','hour']]
y = df['fraud_flag']

print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=7)
model.fit(X, y)

print("💾 Saving new model...")
joblib.dump(model, "fraud_model.joblib")

print("✅ Model retrained successfully! Now you can run fraud_detector.py")
