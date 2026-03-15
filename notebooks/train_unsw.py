import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = r"d:\cyber_attack_detection_project"
train_path = os.path.join(base_dir, 'dataset', 'UNSW_NB15_training-set.parquet')
test_path = os.path.join(base_dir, 'dataset', 'UNSW_NB15_testing-set.parquet')

print("Loading UNSW-NB15 Data...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# Drop the binary 'label' as we will use 'attack_cat' for multiclass classification
if 'label' in df_train.columns:
    df_train.drop(['label'], axis=1, inplace=True)
if 'label' in df_test.columns:
    df_test.drop(['label'], axis=1, inplace=True)

# Standardize attack categories
df_train['attack_cat'] = df_train['attack_cat'].str.lower().str.strip()
df_test['attack_cat'] = df_test['attack_cat'].str.lower().str.strip()
df_train['attack_cat'] = df_train['attack_cat'].replace({'normal': 'normal'})

print("Encoding Categorical Features...")
categoricals = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
if 'attack_cat' in categoricals:
    categoricals.remove('attack_cat')

df_combined = pd.concat([df_train, df_test])
label_encoders = {}

# Fill any NA values for categoricals with 'unknown'
for col in categoricals:
    df_combined[col] = df_combined[col].fillna('unknown')
    le = LabelEncoder()
    df_combined[col] = le.fit_transform(df_combined[col].astype(str))
    label_encoders[col] = le

# Handle numeric NAs
numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
df_combined[numeric_cols] = df_combined[numeric_cols].fillna(0)

train_len = len(df_train)
df_train = df_combined.iloc[:train_len].copy()
df_test = df_combined.iloc[train_len:].copy()

X_train = df_train.drop(['attack_cat'], axis=1)
y_train = df_train['attack_cat']

X_test = df_test.drop(['attack_cat'], axis=1)
y_test = df_test['attack_cat']

print("Training Random Forest Classifier on UNSW-NB15...")
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

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix - UNSW-NB15")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(base_dir, 'notebooks', 'confusion_matrix_unsw.png'))

print("Saving models...")
os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
model_path = os.path.join(base_dir, 'models', 'unsw_model.pkl')
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")

encoders_path = os.path.join(base_dir, 'models', 'unsw_encoders.pkl')
joblib.dump((label_encoders, list(X_train.columns)), encoders_path)
print(f"Encoders saved to {encoders_path}")
print("Training completed successfully!")
