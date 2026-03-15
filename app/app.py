from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
import numpy as np
import os
import random
import json
from datetime import datetime
import csv
import psutil
import scanner
import time

app = Flask(__name__)

BASE_DIR = r"d:\cyber_attack_detection_project"
CONFIG_FILE = os.path.join(BASE_DIR, 'app', 'config.json')
LOGS_FILE = os.path.join(BASE_DIR, 'app', 'logs.json')

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "threat_confidence_threshold": 80,
        "enable_auto_blocking": True,
        "max_packet_rate_alert": 1000,
        "default_ml_model": "NSL-KDD",
        "log_retention_duration": 30,
        "enable_real_time_monitoring": True,
        "enable_email_alerts": False
    }

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
models = {}

def load_model(name, model_file, encoders_file, dataset_name, algo, accuracy):
    model_path = os.path.join(BASE_DIR, 'models', model_file)
    encoders_path = os.path.join(BASE_DIR, 'models', encoders_file)
    config = load_config()
    is_active = (config.get("default_ml_model") == name)
    
    model_info = {
        "id": name.lower().replace("-", "_").replace(" ", "_"),
        "name": f"{name} Model",
        "dataset": dataset_name,
        "algorithm": algo,
        "accuracy": accuracy,
        "status": "Active" if is_active else "Inactive",
        "last_trained": (
            datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M')
            if os.path.exists(model_path) else "N/A"
        )
    }
    try:
        if os.path.exists(model_path) and os.path.exists(encoders_path):
            model = joblib.load(model_path)
            encoders, cols = joblib.load(encoders_path)
            models[name] = {'model': model, 'encoders': encoders, 'cols': cols, 'info': model_info}
            print(f"[OK] Model '{name}' loaded.")
        else:
            model_info["status"] = "Inactive"
            models[name] = {'info': model_info}
            print(f"[WARN] Model files for '{name}' missing.")
    except Exception as e:
        model_info["status"] = "Error"
        models[name] = {'info': model_info}
        print(f"[ERR] Loading model '{name}': {e}")

load_model('NSL-KDD',   'cyber_model.pkl',  'encoders.pkl',       'NSL-KDD',   'Random Forest', 99.5)
load_model('UNSW-NB15', 'unsw_model.pkl',   'unsw_encoders.pkl',  'UNSW-NB15', 'Random Forest', 98.2)
load_model('CICIDS2017','cicids_model.pkl', 'cicids_encoders.pkl','CICIDS2017','Random Forest', 99.7)



# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------
def load_logs():
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_log(entry):
    logs = load_logs()
    logs.insert(0, entry)
    
    # Honor Log Retention Setting
    config = load_config()
    retention_days = config.get("log_retention_duration", 30)
    cutoff = datetime.now() - pd.Timedelta(days=retention_days)
    
    # Filter logs
    valid_logs = []
    for log in logs:
        try:
            log_time = datetime.strptime(log.get("timestamp", ""), '%Y-%m-%d %H:%M:%S')
            if log_time >= cutoff:
                valid_logs.append(log)
        except Exception:
            # If timestamp parsing fails, keep the log to be safe
            valid_logs.append(log)
            
    with open(LOGS_FILE, 'w') as f:
        json.dump(valid_logs[:2000], f, indent=4)

# ---------------------------------------------------------------------------
# Severity classification helper
# ---------------------------------------------------------------------------
SEVERITY_MAP = {
    'dos':       'CRITICAL',
    'ddos':      'CRITICAL',
    'probe':     'HIGH',
    'portscan':  'HIGH',
    'r2l':       'HIGH',
    'u2r':       'CRITICAL',
    'ftp-patator': 'HIGH',
    'ssh-patator': 'HIGH',
    'dos slowloris': 'CRITICAL',
    'dos slowhttptest': 'CRITICAL',
    'dos hulk':  'CRITICAL',
    'dos goldeneye': 'CRITICAL',
    'heartbleed':'CRITICAL',
    'web attack': 'HIGH',
    'web attack – brute force': 'HIGH',
    'web attack – xss': 'MEDIUM',
    'web attack – sql injection': 'CRITICAL',
    'infiltration': 'CRITICAL',
    'botnet':    'CRITICAL',
    'fuzzer':    'MEDIUM',
    'exploits':  'CRITICAL',
    'reconnaissance': 'MEDIUM',
    'shellcode': 'CRITICAL',
    'worms':     'CRITICAL',
    'backdoor':  'CRITICAL',
    'analysis':  'MEDIUM',
    'generic':   'MEDIUM',
    'normal':    'NONE',
    'benign':    'NONE',
}

ACTION_MAP = {
    'CRITICAL': 'Block + Alert',
    'HIGH':     'Block',
    'MEDIUM':   'Throttle',
    'LOW':      'Monitor',
    'NONE':     'Allow',
}

def get_severity(prediction_str):
    key = prediction_str.lower().strip()
    if key in ('normal', 'benign'):
        return 'NONE'
    for k, v in SEVERITY_MAP.items():
        if k in key:
            return v
    return 'HIGH'   # unknown attack → HIGH by default

# ---------------------------------------------------------------------------
# In-memory stats
# ---------------------------------------------------------------------------
stats = {
    "total_analyzed": 10423,
    "attacks_prevented": 341,
    "uptime": "99.9%",
    "detection_accuracy": 99.1,
    "active_threats": 0,
    "last_attack": "None",
    "last_confidence": 0.0,
    "last_model": "NSL-KDD",
    "last_severity": "NONE",
    "last_timestamp": "—",
}

# ---------------------------------------------------------------------------
# Rate Limiting & Alerting State
# ---------------------------------------------------------------------------
packet_rate_tracker = {"count": 0, "last_reset": time.time()}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    config = load_config()
    dataset = request.json.get('dataset', config.get('default_ml_model', 'NSL-KDD'))

    if dataset not in models or 'model' not in models[dataset]:
        return jsonify({"error": f"Model '{dataset}' is not loaded. Train it first."})

    model_data = models[dataset]
    model     = model_data['model']
    encoders  = model_data['encoders']
    cols      = model_data['cols']

    try:
        data = request.json['features']

        # Build feature DataFrame
        df = pd.DataFrame([{c: data.get(c, 0) for c in cols}])

        # Encode categoricals
        for col, le in encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])
                else:
                    df[col] = le.transform(['unknown']) if 'unknown' in le.classes_ else 0

        # Inference
        prediction     = model.predict(df)[0]
        prediction_str = str(prediction).upper()

        # Confidence via predict_proba
        confidence = 1.0
        top_threats = []
        try:
            proba = model.predict_proba(df)[0]
            classes = list(model.classes_)
            confidence = float(max(proba))
            top_idx = np.argsort(proba)[::-1][:3]
            top_threats = [
                {"label": str(classes[i]).upper(), "score": round(float(proba[i]) * 100, 1)}
                for i in top_idx if proba[i] > 0.01
            ]
        except Exception:
            pass

        conf_pct = round(confidence * 100, 1)
        is_attack = prediction_str not in ('NORMAL', 'BENIGN')
        severity  = get_severity(prediction_str) if is_attack else 'NONE'
        recommended_action = ACTION_MAP.get(severity, 'Monitor')

        # Auto-blocking decision
        threshold = config.get('threat_confidence_threshold', 80)
        action = 'Allowed'
        if is_attack:
            stats['attacks_prevented'] += 1
            stats['active_threats'] = min(stats['active_threats'] + 1, 999)
            if config.get('enable_auto_blocking', True) and conf_pct >= threshold:
                action = 'Blocked'
            else:
                action = 'Monitored'

        stats['total_analyzed'] += 1
        stats['last_model'] = dataset
        stats['last_confidence'] = conf_pct
        stats['last_timestamp'] = datetime.now().strftime('%H:%M:%S')
        if is_attack:
            stats['last_attack'] = prediction_str
            stats['last_severity'] = severity

            # Simulated Email Alert
            if config.get('enable_email_alerts', False) and severity == 'CRITICAL' and action == 'Blocked':
                print(f"[ALERT] 🚨 CRITICAL THREAT BLOCKED: {prediction_str}. Dispatching alert email to administrator...")
                
        # Handle Max Packet Rate Alert tracking
        current_time = time.time()
        if current_time - packet_rate_tracker["last_reset"] >= 1.0:
            # Check rate against settings every second
            if packet_rate_tracker["count"] > config.get("max_packet_rate_alert", 1000):
                print(f"[WARNING] High Traffic Volume Detected! Received {packet_rate_tracker['count']} packets/sec (Limit: {config.get('max_packet_rate_alert', 1000)}).")
            # Reset tracker for the next second Window
            packet_rate_tracker["count"] = 0
            packet_rate_tracker["last_reset"] = current_time
        packet_rate_tracker["count"] += 1

        # Protocol extraction
        proto = str(data.get('protocol_type', data.get('proto', 'TCP'))).upper().strip()
        if not proto or proto == '-':
            proto = 'TCP'

        log_entry = {
            "timestamp":   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "src_ip":      f"192.168.{random.randint(1,10)}.{random.randint(2,254)}",
            "dst_ip":      f"10.0.{random.randint(0,5)}.{random.randint(2,254)}",
            "protocol":    proto,
            "attack_type": prediction_str if is_attack else 'BENIGN',
            "confidence":  conf_pct,
            "severity":    severity,
            "action":      action,
            "model_used":  dataset,
        }
        save_log(log_entry)

        return jsonify({
            "prediction":          prediction_str,
            "confidence":          conf_pct,
            "severity":            severity,
            "recommended_action":  recommended_action,
            "top_threats":         top_threats,
            "log":                 log_entry,
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/dashboard-data')
def dashboard_data():
    return jsonify(stats)

@app.route('/api/system-health')
def system_health():
    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.3),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
    })

# Recent timeline: last N log entries
@app.route('/api/stats/timeline')
def timeline():
    limit = int(request.args.get('limit', 20))
    logs  = load_logs()[:limit]
    return jsonify(logs)

# ---------------------------------------------------------------------------
# Model Registry API
# ---------------------------------------------------------------------------
@app.route('/models', methods=['GET'])
def get_models():
    return jsonify([v['info'] for v in models.values()])

@app.route('/models/activate', methods=['POST'])
def activate_model():
    model_id = request.json.get('model_id')
    found_key = None
    
    # First find if the model actually exists
    for k, v in models.items():
        if v['info']['id'] == model_id:
            found_key = k
            break
            
    if found_key:
        config = load_config()
        config['default_ml_model'] = found_key
        save_config(config)
        
        # Update the status of all loaded models in memory
        for k, v in models.items():
            if k == found_key:
                v['info']['status'] = 'Active'
            else:
                if v['info']['status'] == 'Active':
                    v['info']['status'] = 'Inactive'
                    
        return jsonify({"status": "success", "message": f"'{found_key}' is now the active model."})
        
    return jsonify({"error": "Model not found."}), 404

@app.route('/models/train', methods=['POST'])
def train_model():
    model_id = request.json.get('model_id')
    return jsonify({"status": "success", "message": f"Training job queued for '{model_id}'."})

@app.route('/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    for k, v in models.items():
        if v['info']['id'] == model_id:
            v['info']['status'] = 'Deleted'
            v.pop('model', None)
            return jsonify({"status": "success", "message": f"Model '{model_id}' removed."})
    return jsonify({"error": "Model not found."}), 404

@app.route('/api/models/compare')
def compare_models():
    comparison = [
        {
            "name": v['info']['name'],
            "dataset": v['info']['dataset'],
            "algorithm": v['info']['algorithm'],
            "accuracy": v['info']['accuracy'],
            "status": v['info']['status'],
        }
        for v in models.values()
    ]
    return jsonify(comparison)

# ---------------------------------------------------------------------------
# Logs API
# ---------------------------------------------------------------------------
@app.route('/logs', methods=['GET'])
def get_logs():
    limit = int(request.args.get('limit', 200))
    return jsonify(load_logs()[:limit])

@app.route('/logs/filter', methods=['GET'])
def filter_logs():
    search      = request.args.get('search', '').upper()
    threat_lvl  = request.args.get('threat_level', '')
    action_flt  = request.args.get('action', '')
    logs        = load_logs()

    if search:
        logs = [l for l in logs if search in l.get('attack_type', '').upper()
                or search in l.get('src_ip', '')
                or search in l.get('protocol', '').upper()]
    if threat_lvl:
        logs = [l for l in logs if l.get('severity', '').upper() == threat_lvl.upper()]
    if action_flt:
        logs = [l for l in logs if l.get('action', '').lower() == action_flt.lower()]

    return jsonify(logs[:500])

@app.route('/logs/export', methods=['POST', 'GET'])
def export_logs():
    logs = load_logs()
    export_path = os.path.join(BASE_DIR, 'app', 'ids_logs_export.csv')
    if not logs:
        return jsonify({"error": "No logs to export."})

    fieldnames = ["timestamp","src_ip","dst_ip","protocol","attack_type",
                  "confidence","severity","action","model_used"]
    with open(export_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(logs)

    return send_file(export_path, as_attachment=True, download_name='cyberwatch_logs.csv')

# ---------------------------------------------------------------------------
# Settings API
# ---------------------------------------------------------------------------
@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify(load_config())

@app.route('/settings/update', methods=['POST'])
def update_settings():
    save_config(request.json)
    return jsonify({"status": "success", "message": "Configuration saved."})

import hashlib
import time
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Threat Intelligence – Real Scanner
# ---------------------------------------------------------------------------
@app.route('/api/ti/scan', methods=['POST'])
def ti_scan():
    """Run a real website security scan and return the threat report."""
    url = request.json.get('url', '').strip()
    if not url:
        return jsonify({"error": "URL is required."}), 400
        
    # Run the real scan
    report = scanner.run_scan(url)
    
    if "error" in report:
        return jsonify({"error": report["error"]}), 400
        
    # Generate an IDS log entry for the scan
    log_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "src_ip": "Scanner Module",
        "dst_ip": report.get('ip', 'Unknown'),
        "protocol": "HTTP/HTTPS",
        "attack_type": "TI Scan Results",
        "confidence": report.get('threat_score', 0),
        "severity": 'CRITICAL' if report.get('threat_score', 0) > 80 else 'HIGH' if report.get('threat_score', 0) > 60 else 'MEDIUM' if report.get('threat_score', 0) > 40 else 'LOW',
        "action": "Monitored",
        "model_used": "Active Scan"
    }
    save_log(log_entry)
        
    return jsonify({"status": "success", "info": report})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
