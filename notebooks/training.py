import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Column definitions for NSL-KDD
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack","difficulty"]

# Directories
base_dir = r"d:\cyber_attack_detection_project"
train_path = os.path.join(base_dir, 'dataset', 'NSL-KDD', 'KDDTrain+.txt')
test_path = os.path.join(base_dir, 'dataset', 'NSL-KDD', 'KDDTest+.txt')

print("Loading Data...")
df_train = pd.read_csv(train_path, header=None, names=col_names)
df_test = pd.read_csv(test_path, header=None, names=col_names)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# Drop difficulty column (Step 5)
df_train.drop(['difficulty'], axis=1, inplace=True)
df_test.drop(['difficulty'], axis=1, inplace=True)

# Map attacks mapping
dos = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm']
probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
u2r = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
r2l = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']

def map_attack(attack):
    if attack == 'normal': return 'normal'
    if attack in dos: return 'dos'
    if attack in probe: return 'probe'
    if attack in u2r: return 'u2r'
    if attack in r2l: return 'r2l'
    return 'unknown' # In case there are some not covered

df_train['attack_class'] = df_train['attack'].apply(map_attack)
df_test['attack_class'] = df_test['attack'].apply(map_attack)

df_train.drop(['attack'], axis=1, inplace=True)
df_test.drop(['attack'], axis=1, inplace=True)

# Visualizations (EDA)
plt.figure()
sns.countplot(data=df_train, x='attack_class')
plt.title("Attack Distribution - Training Data")
plt.savefig(os.path.join(base_dir, 'notebooks', 'attack_distribution.png'))

print("Encoding Categorical Features...")
# Step 6: Encode Categorical Features
categoricals = ['protocol_type', 'service', 'flag']

# Ensure mapping covers test data by combining them
df_combined = pd.concat([df_train, df_test])
label_encoders = {}
for col in categoricals:
    le = LabelEncoder()
    df_combined[col] = le.fit_transform(df_combined[col])
    label_encoders[col] = le

train_len = len(df_train)
df_train = df_combined.iloc[:train_len].copy()
df_test = df_combined.iloc[train_len:].copy()

# Step 7: Feature and Label Separation
X_train = df_train.drop(['attack_class'], axis=1)
y_train = df_train['attack_class']

X_test = df_test.drop(['attack_class'], axis=1)
y_test = df_test['attack_class']

# Step 9: Train Machine Learning Model
print("Training Random Forest Classifier... (This might take a minute)")
rf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42)
rf.fit(X_train, y_train)

# Step 10: Model Evaluation
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
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(base_dir, 'notebooks', 'confusion_matrix.png'))

print("Saving models...")
# Step 11: Save the Model
os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
model_path = os.path.join(base_dir, 'models', 'cyber_model.pkl')
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")

encoders_path = os.path.join(base_dir, 'models', 'encoders.pkl')
joblib.dump((label_encoders, list(X_train.columns)), encoders_path)
print(f"Encoders saved to {encoders_path}")
print("Training completed successfully!")
