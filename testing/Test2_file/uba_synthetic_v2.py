"""
uba_synthetic_v2.py
----------------------------------------
Generates synthetic UBA (User Behavior Analytics) dataset
aligned with nlp_synthetic_v2.py users and anomaly semantics.

- Normal events: routine logins, emails, file access during work hours.
- Anomalies: late-night access, repeated login failures, high-severity resources.

Output: uba_synthetic_v2.csv
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- CONFIG ---
N_LOGS = 1000
USERS = [
    "phillip.allen@enron.com", "jeff.skilling@enron.com",
    "kenneth.lay@enron.com", "andrew.fastow@enron.com",
    "louise.kitchen@enron.com", "john.lavorato@enron.com"
]
EVENTS = ["login", "file_access", "email_sent", "email_received", "db_query", "logout"]
RESOURCES_NORMAL = ["server01", "intranet", "finance_db", "mailbox", "crm_app"]
RESOURCES_RISKY = ["classified_db", "root_server", "backup_repo", "admin_console"]
DEVICES = ["Windows-PC", "MacBook", "iPhone", "Linux-VM"]
LOCATIONS = ["Houston", "New York", "London", "Remote-VPN"]

ANOMALY_RATE = 0.1
START_DATE = datetime(2024, 1, 1)

# -----------------------------------------------
# --- SYNTHETIC LOG GENERATION ---
# -----------------------------------------------

records = []
for i in range(N_LOGS):
    user = random.choice(USERS)
    ts = START_DATE + timedelta(minutes=random.randint(0, 100000))
    event = random.choice(EVENTS)
    action_status = random.choice(["success", "failure"])
    device = random.choice(DEVICES)
    location = random.choice(LOCATIONS)
    
    is_anomaly = random.random() < ANOMALY_RATE

    # Base resource & severity
    if is_anomaly:
        resource = random.choice(RESOURCES_RISKY)
        # Increase severity for anomalies
        hour = ts.hour
        severity = np.clip(
            np.random.normal(0.7, 0.15)
            + (0.15 if action_status == "failure" else 0)
            + (0.1 if hour < 6 or hour > 22 else 0),
            0, 1
        )
    else:
        resource = random.choice(RESOURCES_NORMAL)
        hour = ts.hour
        severity = np.clip(
            np.random.normal(0.4, 0.15)
            + (0.1 if action_status == "failure" else 0)
            + (0.05 if hour < 6 or hour > 22 else 0),
            0, 1
        )

    # Label anomalies based on combined conditions
    is_anomaly = int(
        (is_anomaly)
        or (action_status == "failure" and severity > 0.7)
        or (ts.hour < 6 or ts.hour > 22 and random.random() < 0.4)
    )

    records.append({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "event_type": event,
        "resource": resource,
        "action_status": action_status,
        "device": device,
        "location": location,
        "severity_score": round(float(severity), 3),
        "is_anomaly": is_anomaly
    })

df = pd.DataFrame(records)

# Ensure reproducibility & some balance
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("uba_synthetic_v2.csv", index=False)

print("✅ Saved UBA synthetic v2 dataset -> uba_synthetic_v2.csv")

# --- Summary ---
print("\nAnomaly Distribution:")
print(df["is_anomaly"].value_counts(normalize=True))
print("\nSample logs:")
print(df.head(5).to_string(index=False))
