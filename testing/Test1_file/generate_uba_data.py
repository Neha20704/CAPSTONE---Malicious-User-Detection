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
RESOURCES = ["server01", "intranet", "finance_db", "mailbox", "crm_app"]
DEVICES = ["Windows-PC", "MacBook", "iPhone", "Linux-VM"]
LOCATIONS = ["Houston", "New York", "London", "Remote-VPN"]

# --- Generate system logs (system_logs.csv) ---
data = []
start_date = datetime(2024, 1, 1)

for i in range(N_LOGS):
    user = random.choice(USERS)
    ts = start_date + timedelta(minutes=random.randint(0, 100000))
    event = random.choice(EVENTS)
    resource = random.choice(RESOURCES)
    action_status = random.choice(["success", "failure"])
    device = random.choice(DEVICES)
    location = random.choice(LOCATIONS)
    
    # heuristic for severity: failure events or odd hours = higher severity
    hour = ts.hour
    severity = np.clip(
        np.random.normal(0.5, 0.2)
        + (0.2 if action_status == "failure" else 0)
        + (0.1 if hour < 6 or hour > 22 else 0),
        0, 1
    )

    data.append({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "event_type": event,
        "resource": resource,
        "action_status": action_status,
        "device": device,
        "location": location,
        "severity_score": round(float(severity), 3)
    })

df = pd.DataFrame(data)
df.to_csv("uba_test1.csv", index=False)
print("✅ Saved synthetic UBA data -> uba_test1.csv")
