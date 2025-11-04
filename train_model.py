import pandas as pd
import joblib

def predict_on_new(data):
    model = joblib.load("fraud_model.joblib")
    preds = model.predict(data)

    # Add ML prediction
    data["ml_prediction"] = preds

    # RULE-BASED FRAUD LOGIC (Hybrid System)
    data["rule_flag"] = data.apply(lambda row: 1 if (row["amount"] > 20000 and row["location_score"] < 0.3) else 0, axis=1)

    # Final fraud flag = ML or Rule
    data["fraud_prediction"] = data.apply(lambda row: 1 if (row["ml_prediction"] == 1 or row["rule_flag"] == 1) else 0, axis=1)

    data.to_csv("predictions.csv", index=False)
    print("✅ Fraud detection completed. Check predictions.csv file.")

if __name__ == "__main__":
    df = pd.read_csv("new_transactions.csv")
    predict_on_new(df)
