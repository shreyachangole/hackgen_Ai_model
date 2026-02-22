import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

# =============================
# 1️⃣ Load Data
# =============================
df = pd.read_csv("simulated_50_bins.csv")

# Time features
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

# 🎯 Target
df["will_be_full_24hrs"] = df["hours_to_full"] <= 24

# =============================
# 2️⃣ Feature Engineering
# =============================

# Encode location
le = LabelEncoder()
df["location_type"] = le.fit_transform(df["location_type"])

# Final Features (IMPORTANT ⚠️ same everywhere use kar)
features = [
    "fill_percent",
    "location_type",
    "is_weekend",
    "hour"
]

X = df[features]
y = df["will_be_full_24hrs"]

# =============================
# 3️⃣ Train Model
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight={0: 2, 1: 1},  # reduce false positives
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model Trained")

# =============================
# 4️⃣ Evaluation
# =============================
proba = model.predict_proba(X_test)[:, 1]

threshold = 0.7
y_pred = (proba > threshold)

print("\n📊 Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =============================
# 5️⃣ Save Model
# =============================
joblib.dump(model, "waste_collection_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n✅ Model Saved")

# =============================
# 6️⃣ LIVE BIN STATUS (IMPORTANT)
# =============================

# Latest record per bin
latest_data = df.sort_values("timestamp").groupby("bin_id").tail(1).copy()

# Ensure same features
X_live = latest_data[features]

# Prediction
proba_live = model.predict_proba(X_live)[:, 1]

threshold = 0.7
latest_data["collection_required"] = proba_live > threshold

# Priority (🔥 upgrade)
latest_data["priority"] = proba_live

# =============================
# 7️⃣ FINAL OUTPUT
# =============================

collection_bins = latest_data[latest_data["collection_required"] == True]

# Sort by priority
collection_bins = collection_bins.sort_values(by="priority", ascending=False)

print("\n📦 Total Bins:", len(latest_data))
print("🚛 Need Collection:", len(collection_bins))

print("\n🔴 Bin IDs:")
print(collection_bins["bin_id"].values)

print("\n⭐ Priority:")
print(collection_bins[["bin_id", "priority"]])