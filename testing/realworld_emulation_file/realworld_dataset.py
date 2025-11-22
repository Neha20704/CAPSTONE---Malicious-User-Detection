import pandas as pd
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
np.random.seed(42)
num_users = 20
emails_per_user = np.random.randint(5, 15, num_users)
logs_per_user = np.random.randint(8, 20, num_users)
users = [f"user{i}@enron.com" for i in range(num_users)]

# ---------------------------
# NLP DATASET (Email-like)
# ---------------------------
nlp_rows = []
for user, n in zip(users, emails_per_user):
    for i in range(n):
        subject = np.random.choice([
            "Quarterly report", "Server issue", "Meeting schedule",
            "Expense review", "Access request", "Password reset",
            "Data access request", "Incident update"
        ])
        msg_body = np.random.choice([
            "please review the attached file",
            "server access denied for some users",
            "kindly approve the budget request",
            "new updates to compliance policy",
            "urgent: data sharing request from external domain",
            "internal audit scheduled for next week"
        ])
        
        # Create realistic multi-line raw email
        raw_msg = f"""Message-ID: <{np.random.randint(100000,999999)}.{i}@enron.com>
Date: Mon, {np.random.randint(1, 28)} May 2001 {np.random.randint(8, 19)}:{np.random.randint(10, 59):02d}:00 -0700 (PDT)
From: {user}
To: {np.random.choice(users)}
Subject: {subject}
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: {user.split('@')[0].title()}
X-To: {np.random.choice(users)}
X-cc: 
X-bcc: 
X-Folder: \\{user.split('@')[0].title()}\\Sent
X-Origin: {user.split('@')[0].title()}
X-FileName: {user.split('@')[0].lower()}.pst

{msg_body}
"""
        nlp_rows.append({
            "file": f"mail_{user}_{i}.txt",
            "from": user,
            "to": np.random.choice(users),
            "subject": subject,
            "cleaned_message": msg_body.lower(),
            "message": raw_msg,   # Raw Enron-like message
            "anomaly_score": np.clip(np.random.normal(0.4, 0.2), 0, 1)
        })

nlp_df = pd.DataFrame(nlp_rows)

# ---------------------------
# UBA DATASET (Log-like)
# ---------------------------
uba_rows = []
for user, n in zip(users, logs_per_user):
    for i in range(n):
        uba_rows.append({
            "timestamp": pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 60)),
            "user": user,
            "event_type": np.random.choice(["login", "file_access", "permission_change", "email_send"]),
            "resource": np.random.choice(["server1", "finance_db", "email_gateway"]),
            "action_status": np.random.choice(["success", "fail"]),
            "device": np.random.choice(["laptop", "mobile", "desktop"]),
            "location": np.random.choice(["US", "EU", "IN"]),
            "severity_score": np.clip(np.random.normal(0.5, 0.25), 0, 1),
            "final_risk": np.clip(np.random.normal(0.5, 0.2), 0, 1)
        })

uba_df = pd.DataFrame(uba_rows)

# ---------------------------
# SAVE TO CSV
# ---------------------------
nlp_df.to_csv("synthetic_nlp_realworld.csv", index=False)
uba_df.to_csv("synthetic_uba_realworld.csv", index=False)

print(f"✅ Created NLP ({len(nlp_df)}) and UBA ({len(uba_df)}) test data for {len(users)} users.")
print("📂 Files: synthetic_nlp_realworld.csv, synthetic_uba_realworld.csv")
