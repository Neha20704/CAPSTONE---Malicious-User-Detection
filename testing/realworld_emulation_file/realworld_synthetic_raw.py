import pandas as pd
import numpy as np

np.random.seed(7)

rows = []
users = [f"user{i}@enron.com" for i in range(10)]

for u in users:
    for i in range(3):  # 3 emails each
        msg = f"""Message-ID: <{np.random.randint(100000,999999)}.{i}@enron.com>
Date: Tue, {np.random.randint(1, 28)} May 2001 {np.random.randint(8, 19)}:{np.random.randint(10, 59):02d}:00 -0700 (PDT)
From: {u}
To: {np.random.choice(users)}
Subject: {np.random.choice(['Project Update', 'Server Issue', 'Meeting Request', 'Audit Notice'])}
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: {u.split('@')[0].title()}
X-To: {np.random.choice(users)}
X-Folder: \\{u.split('@')[0].title()}\\Sent
X-Origin: {u.split('@')[0].title()}
X-FileName: {u.split('@')[0].lower()}.pst

{np.random.choice([
    'Please review the attached report.',
    'Our meeting has been rescheduled.',
    'Access to the financial server is restricted.',
    'The system will be offline tonight for maintenance.'
])}
"""
        rows.append({
            "file": f"{u.split('@')[0]}_sent_mail_{i}.txt",
            "message": msg
        })

df = pd.DataFrame(rows)
df.to_csv("synthetic_enron_raw.csv", index=False)
print("✅ Saved raw Enron-style file: synthetic_enron_raw.csv")
