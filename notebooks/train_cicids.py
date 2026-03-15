import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = r"d:\cyber_attack_detection_project"
dataset_dir = os.path.join(base_dir, 'dataset', 'MachineLearningCVE')

print("Loading CICIDS2017 Data (Sampling to reduce memory load)...")
csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))

df_list = []
for file in csv_files:
    # Read and sample 10% of each file to avoid memory issues
    df_part = pd.read_csv(file)
    df_part = df_part.sample(frac=0.1, random_state=42)
    df_list.append(df_part)

df = pd.concat(df_list, ignore_index=True)
print("Combined shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={'label': 'attack_cat'}, inplace=True)

# Replace Infinity and NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Encoding Categorical Features...")
categoricals = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'attack_cat' in categoricals:
    categoricals.remove('attack_cat')

label_encoders = {}
for col in categoricals:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separate features and labels
X = df.drop(['attack_cat'], axis=1)
y = df['attack_cat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Random Forest Classifier on CICIDS2017...")
rf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Evaluating model...")
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save the confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix - CICIDS2017")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(base_dir, 'notebooks', 'confusion_matrix_cicids.png'))

print("Saving models...")
os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
model_path = os.path.join(base_dir, 'models', 'cicids_model.pkl')
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")

encoders_path = os.path.join(base_dir, 'models', 'cicids_encoders.pkl')
joblib.dump((label_encoders, list(X_train.columns)), encoders_path)
print(f"Encoders saved to {encoders_path}")
print("Training completed successfully!")
