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
KEYWORDS = ["confidential", "breach", "server", "meeting", "unauthorized",
            "project", "credentials", "finance", "urgent", "access"]

# --- Generate raw emails (emails.csv) ---
raw_data = []
for i in range(N_EMAILS):
    sender = random.choice(USERS)
    recipient = random.choice([u for u in USERS if u != sender])
    cc = random.choice([random.choice(USERS), ""])  # sometimes empty
    bcc = random.choice([random.choice(USERS), ""])
    subject = random.choice(["Project Update", "Security Notice", "Access Report", "Finance Summary"])
    msg_body = f"This is a {random.choice(KEYWORDS)} email for internal communication."

    # Random-ish date
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
        "subject": subject
    })
emails_df = pd.DataFrame(raw_data)
emails_df.to_csv("test1_emails.csv", index=False)
print("✅ Saved raw test data -> test1_emails.csv")

# --- Generate cleaned / enriched version (enron_recleaned.csv) ---
cleaned_data = []
start_date = datetime(2024, 1, 1)

for i in range(N_EMAILS):
    sender = random.choice(USERS)
    to_list = random.sample(USERS, k=random.randint(1, 3))
    cc_list = random.sample([u for u in USERS if u not in to_list], k=random.randint(0, 2))
    bcc_list = random.sample([u for u in USERS if u not in to_list + cc_list], k=random.randint(0, 1))
    date = (start_date + timedelta(hours=random.randint(0, 5000))).strftime("%Y-%m-%d %H:%M:%S")

    cleaned_data.append({
        "file": f"email_{i}.txt",
        "message": f"Hello team, please note: {random.choice(KEYWORDS)} discussion ahead.",
        "from": sender,
        "to": ", ".join(to_list),
        "cc": ", ".join(cc_list),
        "bcc": ", ".join(bcc_list),
        "date": date,
        "subject": f"{random.choice(['Project', 'Finance', 'Security'])} Update",
        "cleaned_message": f"cleaned text about {random.choice(KEYWORDS)}"
    })

cleaned_df = pd.DataFrame(cleaned_data)
cleaned_df.to_csv("nlp_test1_cleaned.csv", index=False)
print("✅ Saved enriched test data -> nlp_test1_cleaned.csv")
