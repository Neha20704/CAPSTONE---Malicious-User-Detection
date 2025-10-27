# generate_system_logs.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Load communication dataset
df = pd.read_csv("enron_recleaned.csv")  # path to your current email data

# Extract unique users from the 'from' field
users = df["from"].dropna().unique()

# Possible event types
events = [
    "login_success",
    "login_failure",
    "file_access",
    "file_delete",
    "privilege_escalation",
    "data_exfiltration",
    "usb_insert",
    "network_connection"
]

system_logs = []

for user in users:
    for _ in range(random.randint(20, 50)):  # 20–50 events per user
        log = {
            "timestamp": datetime(2000, 1, 1) + timedelta(days=random.randint(0, 800)),
            "user": user,
            "event_type": random.choice(events),
            "resource": random.choice(["finance.xls", "hr.db", "contracts.pdf", "server01", "usb_drive"]),
            "action_status": random.choice(["success", "failure"]),
            "device": random.choice(["desktop", "vpn", "mobile"]),
            "location": random.choice(["Houston", "London", "New York", "Remote", "Tokyo"]),
            "severity_score": round(random.uniform(0.1, 1.0), 2)
        }
        system_logs.append(log)

# Create DataFrame and save
system_df = pd.DataFrame(system_logs)
system_df.to_csv("system_logs.csv", index=False)
print("✅ Generated synthetic system_logs.csv")
