import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("soil_recommendations.csv")

# Features & target
X = df[["soil_moisture", "soil_type", "region"]]
y = df["improvement_step"]

# Initialize encoders
le_soil = LabelEncoder()
le_region = LabelEncoder()
le_target = LabelEncoder()

# Fit encoders on FULL dataset (not only train split!)
X["soil_type"] = le_soil.fit_transform(X["soil_type"])
X["region"] = le_region.fit_transform(X["region"])
y = le_target.fit_transform(y)

# Save the classes for future unseen inputs
joblib.dump(le_soil, "encoder_soil.pkl")
joblib.dump(le_region, "encoder_region.pkl")
joblib.dump(le_target, "encoder_target.pkl")

# Now split AFTER encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("✅ Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "soil_quality_model.pkl")
