
# 🚨 Fake Job Postings Detection using TF-IDF, Logistic Regression & SVM

## 📌 Overview

Online recruitment platforms have become essential for job seekers. However, the rise of fraudulent job postings has exposed applicants to scams, identity theft, and financial exploitation.

This project presents a **Machine Learning–based Fake Job Detection System** that automatically classifies job postings as *Real* or *Fraudulent* using Natural Language Processing (NLP) techniques.

The system leverages **TF-IDF vectorization** for feature extraction and compares two classical machine learning models:

* Logistic Regression (LR)
* Support Vector Machine (SVM)

The goal is to build an interpretable and high-performing fraud detection model that can enhance trust in digital hiring platforms.

---

## 🎯 Problem Statement

Fake job advertisements:

* Exploit unemployed individuals
* Collect illegal fees
* Steal personal data
* Damage trust in online recruitment systems

Manual verification is inefficient and not scalable.

This project develops an automated fraud detection pipeline capable of identifying deceptive job postings based on textual patterns.

---

## 🎯 Objectives

* Detect fraudulent job postings using text classification
* Implement TF-IDF for feature engineering
* Compare Logistic Regression and SVM performance
* Evaluate models using Accuracy, Precision, Recall, and F1-score
* Build a reproducible and structured ML pipeline

---

## 📂 Project Structure

'''
fake-job-detection/
│
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks (experiments & analysis)
├── src/                   # Preprocessing & modeling scripts
├── models/                # Saved trained models
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
'''

---

## 📊 Dataset Information

* **Type:** Text-based recruitment dataset

* **Target Variable:** Fraudulent (0 = Real, 1 = Fake)

* **Features Used:**

  * job_description
  * title
  * company_profile (where applicable)

* **Data Split:**

  * 80% Training
  * 10% Validation
  * 10% Testing

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

* Text cleaning
* Lowercasing
* Stopword removal
* Removal of special characters
* Tokenization

---

### 2️⃣ Feature Engineering

* **TF-IDF (Term Frequency–Inverse Document Frequency)**

  * Converts text into numerical vectors
  * Handles high-dimensional sparse data
  * Captures important textual signals

---

### 3️⃣ Models Implemented

#### 🔹 Logistic Regression

* Linear classifier
* Interpretable
* Strong baseline model

#### 🔹 Support Vector Machine (SVM)

* Effective in high-dimensional text classification
* Maximizes margin between classes
* Strong generalization performance

---

### 4️⃣ Evaluation Metrics

To ensure balanced fraud detection performance, models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

These metrics help minimize false positives and false negatives.

---

## 📈 Results

The models were evaluated using Accuracy, Precision, Recall, and F1-score to ensure balanced performance assessment, particularly due to class imbalance in the dataset (significantly more real jobs than fraudulent ones).

---

### 🔹 Logistic Regression (Baseline Model)

**Validation Accuracy:** 98.32%
**Test Accuracy:** 98.83%

**Test Classification Report:**

| Class    | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Real (0) | 0.99      | 1.00   | 0.99     |
| Fake (1) | 0.99      | 0.77   | 0.86     |

#### Key Observations

* The model demonstrates near-perfect detection of legitimate job postings.
* Fraudulent postings are identified with **99% precision**, meaning when the model flags a job as fake, it is almost always correct.
* Recall for fraudulent jobs is **77%**, indicating that most fake postings are detected, though some remain challenging due to dataset imbalance.

---

### 🔹 Support Vector Machine (SVM) Results

The SVM model demonstrated competitive performance in high-dimensional TF-IDF feature space, maintaining strong classification boundaries and generalization capability.

Comparative evaluation showed that both Logistic Regression and SVM perform strongly for text-based fraud detection, with SVM providing robust separation in feature space.

---

### 🔹 Ensemble (Bagging) Performance

Bagging approaches were tested to improve robustness; however:

* Logistic Regression with bagging reduced fraud recall significantly.
* SVM with bagging showed moderate performance but did not outperform the standalone models.

This indicates that the standalone classifiers were better suited for this specific dataset and feature representation.

---

## 🔎 Overall Performance Insight

* Final Test Accuracy: **98.83%**
* Fraud Detection Precision: **99%**
* Fraud Detection F1-Score: **0.86**

The model achieves strong overall classification performance while maintaining high precision for fraudulent postings — a critical factor in fraud detection systems where false accusations must be minimized.

The results demonstrate that classical machine learning models combined with TF-IDF feature extraction remain highly effective for structured text-based fraud detection tasks.

---

## 🔍 Key Findings

* TF-IDF effectively captures fraudulent linguistic patterns
* SVM typically performs strongly in high-dimensional text problems
* Logistic Regression provides interpretability and competitive baseline performance
* Class imbalance significantly affects fraud detection systems and must be carefully handled

---

## 🛠 Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Jupyter Notebook

---

## 🚀 Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/fake-job-detection.git
cd fake-job-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the notebook

Open Jupyter Notebook and run the main experiment file inside the `notebooks/` folder.

---

## 📌 Future Improvements

* Deploy as a REST API using Flask or FastAPI
* Build a web interface for recruiters
* Try deep learning models (LSTM / Transformers)
* Apply advanced imbalance handling techniques
* Integrate explainability tools (e.g., LIME or SHAP)

---

## 🌍 Potential Applications

* Online recruitment platforms
* HR technology systems
* Job listing websites
* Fraud monitoring systems

---

## 👤 Author

**Ogundolire Leye Bayowa**
Machine Learning Engineer | Data Analyst |

Passionate about building data-driven systems that solve real societal problems.

---

# 🔎 GitHub Optimization Tips (Important)

To improve visibility:

### ✅ Add Topics/Tags on GitHub

```
machine-learning
natural-language-processing
fraud-detection
text-classification
tfidf
svm
logistic-regression
python
data-science
```

### ✅ Repository Name Suggestion

```
fake-job-postings-detection-ml
```

### ✅ Add

* A clean project banner (optional)
* Clear commit history
* Proper requirements.txt
* A short project description under repo title

---
