# Capstone: Malicious User Detection

#  Malicious User Detection (Capstone Project)

This project detects **insider threats** and **malicious email behavior** using the Enron email dataset.
It combines **machine learning models**, **NLP preprocessing**, and a **Streamlit web app** for interactive anomaly analysis.

---

## Features

* **Email Preprocessing & Feature Engineering**

  * Text cleaning, tokenization
  * TF-IDF vectorization
  * Sentiment polarity extraction
  * Communication pattern features (to/cc/bcc counts, off-hours detection, etc.)
  * Threat keyword tagging

* **Machine Learning Models**

  * Isolation Forest
  * One-Class SVM
  * Autoencoder (Keras)

* **Anomaly Detection**

  * Assigns anomaly scores to each email
  * Detects unusual communication patterns
  * Flags potential insider threats

* **Interactive Dashboard (Streamlit)**

  * Upload raw email CSV
  * Automatic feature extraction
  * Model-based anomaly predictions
  * Visualization of results and threat indicators

---

## Project Structure

```
.
├── data/                # Dataset files (ignored in Git)
│   └── raw/             # Raw Enron emails
│   └── processed/       # Preprocessed data
│
├── models/              # Trained ML models (.pkl, .keras, etc.)
│   └── models_1/        # Traditional ML (Isolation Forest, RF, etc.)
│   └── models_2/        # Deep learning + hybrids
│
├── notebooks/           # Jupyter notebooks for training & EDA
│
├── src/                 
│   ├── app.py           # Streamlit dashboard
│   ├── load_models.py   # Utility functions to load models
│   ├── train_local.py   # Local training script
│   └── preprocessing.py # Text cleaning & feature extraction
│
├── .gitignore
├── .gitattributes       # Git LFS tracking for large files
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1 Clone the repo

```bash
git clone https://github.com/Neha20704/CAPSTONE-Malicious-User-Detection.git
cd CAPSTONE-Malicious-User-Detection
```

### 2️ Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

### 3️ Run the Streamlit app

```bash
streamlit run src/app.py
```

---

## How It Works

1. **Upload raw email CSV** into the app.
2. **Feature extraction** (text cleaning, NLP, sentiment, metadata).
3. **Models run predictions** (Isolation Forest, OCSVM, Autoencoder, etc.).
4. **Results shown on dashboard** with anomaly scores & visualizations.

---

##  Models Used

* **Isolation Forest** → detects outliers based on communication patterns.
* **One-Class SVM** → identifies deviations from normal behavior.
* **Autoencoder (Keras)** → reconstructs normal email patterns, high error = anomaly.
* **Random Forest** → supervised baseline model.

---

## Example Output

*  Normal email → low anomaly score
* Suspicious email → flagged by one or more models
*  Malicious email → high anomaly score + multiple threat indicators

---

##  Team
Developed by:
**Naru Meghana PES2UG22CS341**
**Neha Girish PES2UG22CS346**
**Nikitha Thammaiah PES2UG22CS362**
**Shivprakash G PES2UG23CS822**

---
