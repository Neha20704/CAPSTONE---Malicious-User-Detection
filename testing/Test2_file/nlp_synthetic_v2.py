"""
nlp_synthetic_v2.py
----------------------------------------
Generates two aligned synthetic Enron-style email datasets:
1️⃣ Raw email text (with headers) → nlp_synthetic_v2_raw.csv
2️⃣ Cleaned/enriched version → nlp_synthetic_v2_cleaned.csv
Includes 10% anomalous emails (with risky terms) for testing.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- CONFIG ---
N_EMAILS = 500
USERS = [
    "phillip.allen@enron.com", "jeff.skilling@enron.com",
    "kenneth.lay@enron.com", "andrew.fastow@enron.com",
    "louise.kitchen@enron.com", "john.lavorato@enron.com"
]
KEYWORDS_SAFE = ["project", "meeting", "update", "report", "schedule", "discussion"]
KEYWORDS_RISKY = ["confidential", "password", "transfer", "unauthorized", "breach", "server", "leak"]
ANOMALY_RATE = 0.1  # 10% anomalous
START_DATE = datetime(2024, 1, 1)

# -----------------------------------------------
# --- RAW EMAIL GENERATION ---
# -----------------------------------------------

raw_data = []
for i in range(N_EMAILS):
    sender = random.choice(USERS)
    recipient = random.choice([u for u in USERS if u != sender])
    cc = random.choice([random.choice(USERS), ""])
    bcc = random.choice([random.choice(USERS), ""])
    is_anomaly = random.random() < ANOMALY_RATE

    if is_anomaly:
        keyword = random.choice(KEYWORDS_RISKY)
        subject = f"URGENT: {keyword.title()} Detected"
        msg_body = f"This email discusses {keyword} information. Handle with caution and delete after reading."
    else:
        keyword = random.choice(KEYWORDS_SAFE)
        subject = f"Weekly {keyword.title()} Update"
        msg_body = f"Regular internal communication regarding {keyword} tasks and progress."

    # Random realistic date
    date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%a, %d %b %Y %H:%M:%S -0800")

    # Construct the full raw email text
    msg = (
        f"From: {sender}\n"
        f"To: {recipient}\n"
        f"Cc: {cc}\n"
        f"Bcc: {bcc}\n"
        f"Date: {date}\n"
        f"Subject: {subject}\n\n"
        f"{msg_body}"
    )

    raw_data.append({
        "file": f"email_{i}.txt",
        "message": msg,
        "from": sender,
        "to": recipient,
        "cc": cc,
        "bcc": bcc,
        "date": date,
        "subject": subject,
        "is_anomaly": int(is_anomaly)
    })

raw_df = pd.DataFrame(raw_data)
raw_df.to_csv("nlp_synthetic_v2_raw.csv", index=False)
print("✅ Saved raw synthetic emails -> nlp_synthetic_v2_raw.csv")

# -----------------------------------------------
# --- CLEANED / ENRICHED VERSION ---
# -----------------------------------------------

cleaned_data = []
for i in range(N_EMAILS):
    sender = raw_df.loc[i, "from"]
    to_list = [raw_df.loc[i, "to"]]
    cc_list = [raw_df.loc[i, "cc"]] if raw_df.loc[i, "cc"] else []
    bcc_list = [raw_df.loc[i, "bcc"]] if raw_df.loc[i, "bcc"] else []
    date = (START_DATE + timedelta(hours=random.randint(0, 5000))).strftime("%Y-%m-%d %H:%M:%S")
    is_anomaly = raw_df.loc[i, "is_anomaly"]

    if is_anomaly:
        keyword = random.choice(KEYWORDS_RISKY)
        cleaned_message = f"alert: contains sensitive {keyword} data, monitor activity"
    else:
        keyword = random.choice(KEYWORDS_SAFE)
        cleaned_message = f"routine communication about {keyword}"

    cleaned_data.append({
        "file": f"email_{i}.txt",
        "message": f"Hello team, please note: {keyword} discussion ahead.",
        "from": sender,
        "to": ", ".join(to_list),
        "cc": ", ".join(cc_list),
        "bcc": ", ".join(bcc_list),
        "date": date,
        "subject": raw_df.loc[i, "subject"],
        "cleaned_message": cleaned_message,
        "is_anomaly": int(is_anomaly)
    })

cleaned_df = pd.DataFrame(cleaned_data)
cleaned_df.to_csv("nlp_synthetic_v2_cleaned.csv", index=False)
print("✅ Saved enriched synthetic data -> nlp_synthetic_v2_cleaned.csv")

# --- Summary ---
print("\nAnomaly Distribution:")
print(cleaned_df["is_anomaly"].value_counts(normalize=True))
print("\nSample message preview:\n")
print(cleaned_df["message"].iloc[0][:400], "...")
