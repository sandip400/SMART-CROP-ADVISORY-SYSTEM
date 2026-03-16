import joblib
import pandas as pd

# ----------------------
# Load model + encoders
# ----------------------
model = joblib.load("soil_quality_model.pkl")
le_soil = joblib.load("encoder_soil.pkl")
le_region = joblib.load("encoder_region.pkl")
le_target = joblib.load("encoder_target.pkl")

# ----------------------
# Safe encoding function
# ----------------------
def safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        print(f"⚠️ Warning: unseen label '{value}', assigning default class {le.classes_[0]}")
        return 0  # fallback to first class

# ----------------------
# Example user input
# ----------------------
soil_type = "clay"      # e.g., sand / clay / loam / silty
region = "XMS-CAT"      # must match what was in dataset, else fallback
moisture = 45.0         # soil moisture percentage

# ----------------------
# Encode input
# ----------------------
X_new = pd.DataFrame([{
    "soil_moisture": moisture,
    "soil_type": safe_encode(le_soil, soil_type),
    "region": safe_encode(le_region, region)
}])

# ----------------------
# Predict
# ----------------------
pred = model.predict(X_new)[0]
recommendation = le_target.inverse_transform([pred])[0]

print("🌱 Soil Type:", soil_type)
print("📍 Region:", region)
print("💧 Moisture:", moisture)
print("🪴 Recommended Step:", recommendation)
