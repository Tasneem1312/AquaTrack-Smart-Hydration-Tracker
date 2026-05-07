import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# ================================================================
# GENERATE SYNTHETIC HYDRATION TRAINING DATA
# Mimics real user drinking patterns over many days
# ================================================================
np.random.seed(42)
N = 3000  # training samples

def generate_dataset(n):
    
    rows = []
    for _ in range(n):
        # Feature 1: minutes since last drink (0 - 120)
        mins_since   = np.random.exponential(scale=35) 
        mins_since   = np.clip(mins_since, 0, 120)

        # Feature 2: hour of day (0-23)
        hour         = np.random.randint(6, 23)

        # Feature 3: drinks so far today (0-15)
        drinks_today = np.random.randint(0, 15)

        # Feature 4: average interval between drinks today (mins)
        avg_interval = np.random.normal(40, 15)
        avg_interval = np.clip(avg_interval, 10, 90)

        # Feature 5: activity level proxy — tilt events last 10 min
        activity     = np.random.randint(0, 8)

        # Feature 6: time of day bucket
        # Morning(6-10)=1, Midday(10-14)=2, Afternoon(14-18)=3, Evening(18-23)=4
        if   6  <= hour < 10: tod = 1
        elif 10 <= hour < 14: tod = 2
        elif 14 <= hour < 18: tod = 3
        else:                 tod = 4

        # ── Label: will user drink in next 10 minutes? ────────────
        # Probability increases with time since last drink,
        # decreases if they drank recently, varies by time of day
        base_prob = mins_since / 60.0  # 0 to 2.0

        # More likely to drink at meal times
        tod_boost = {1: 0.3, 2: 0.5, 3: 0.2, 4: 0.3}.get(tod, 0.2)

        # Less likely if they just drank
        recent_penalty = max(0, 0.5 - mins_since / 30.0)

        # Activity increases thirst
        activity_boost = activity * 0.05

        prob = base_prob + tod_boost + activity_boost - recent_penalty
        prob = np.clip(prob, 0.05, 0.95)

        # Add noise
        prob += np.random.normal(0, 0.1)
        prob  = np.clip(prob, 0, 1)

        label = 1 if prob > 0.55 else 0

        rows.append([mins_since, hour, drinks_today,
                     avg_interval, activity, tod, label])

    df = pd.DataFrame(rows, columns=[
        'mins_since_last', 'hour_of_day', 'drinks_today',
        'avg_interval', 'activity_level', 'time_of_day_bucket', 'label'
    ])
    return df

print("Generating training data...")
df = generate_dataset(N)
# Save dataset to CSV so you can show it
df.to_csv('hydration_dataset.csv', index=False)
print(f"Dataset saved to hydration_dataset.csv")
print(f"Dataset: {len(df)} samples")
print(f"Class balance: {df['label'].value_counts().to_dict()}")

# ================================================================
# TRAIN MODEL
# ================================================================
features = ['mins_since_last', 'hour_of_day', 'drinks_today',
            'avg_interval', 'activity_level', 'time_of_day_bucket']
X = df[features].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\nTraining Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    random_state=42
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
acc    = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"{'='*40}")
print(classification_report(y_test, y_pred,
      target_names=['No drink soon', 'Drink predicted']))

# ================================================================
# SAVE MODEL + SCALER
# ================================================================
joblib.dump(model,  'hydration_model.pkl')
joblib.dump(scaler, 'hydration_scaler.pkl')

# Save feature names for dashboard
meta = {
    'features':  features,
    'accuracy':  round(acc * 100, 2),
    'n_samples': N,
    'model':     'GradientBoostingClassifier (TinyML proxy)'
}
with open('model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nSaved: hydration_model.pkl")
print("Saved: hydration_scaler.pkl")
print("Saved: model_meta.json")
print("\nDone! Use these files in your dashboard.")
