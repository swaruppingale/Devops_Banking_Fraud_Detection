import pandas as pd
import joblib
import sys

MODEL_PATH = "fraud_model.joblib"
DATA_PATH = "transactions.csv"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_on_new(dataframe):
    features = ["amount","frequency","location_score","hour"]
    model = load_model()
    preds = model.predict(dataframe[features])
    return preds

def main():
    # If a CSV path is passed, use that; else use transactions.csv
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    df = pd.read_csv(csv_path)
    preds = predict_on_new(df)
    out = df.copy()
    out["predicted_fraud"] = preds
    out.to_csv("predictions.csv", index=False)
    suspicious = out[out["predicted_fraud"] == 1]
    if len(suspicious) > 0:
        print("⚠ Suspicious transactions detected:", len(suspicious))
    else:
        print("✅ No suspicious transactions detected.")
    print("Saved predictions to predictions.csv")

if __name__ == "__main__":
    main()
