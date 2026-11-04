# 🎓 Student Placement Prediction & Performance Analysis

An end-to-end Machine Learning classification pipeline trained on 10,000 student records to predict campus placement outcomes (`Placed` vs. `Not Placed`) based on academic and behavioral drivers. Developed with Scikit-learn classification algorithms and Decision Tree interpretability.

---

## 📌 Project Overview

Campus placements are critical milestones for both students and educational institutions. This project analyzes behavioral patterns—such as study hours, class attendance, assignment completion, and examination scores—to build classification models capable of predicting placement outcomes. 

The primary goal is to provide actionable early-warning insights so mentors and educators can identify at-risk students and intervene proactively.

---

## 📊 Dataset Details

- **Dataset File:** `student_dataset_10000_rows.csv`
- **Total Records:** 10,000 rows, 8 columns
- **Missing Values:** 0 null values across all features
- **Target Feature:** `placement_status` (`Placed` / `Not Placed`)
- **Key Attributes:**
  - `study_hours`: Daily self-study duration (hours)
  - `attendance`: Class attendance percentage (%)
  - `sleep_hours`: Average daily sleep duration (hours)
  - `internet_usage`: Daily recreational/academic internet usage (hours)
  - `assignments_completed`: Count of submitted assignments
  - `previous_score`: Historical academic assessment percentage
  - `exam_score`: Qualifying semester exam score

---

## ⚙️ Technical Workflow

### 1. Exploratory Data Analysis (EDA)
- Validated dataset completeness and verified zero null/missing entries.
- Profiled statistical distributions, correlations, and target class distributions.

### 2. Preprocessing & Feature Scaling
- Encoded categorical target labels (`Placed` / `Not Placed`) into numerical representations.
- Scaled continuous numerical attributes using Scikit-learn's `StandardScaler` to ensure optimal gradient convergence and fair metric weighting.
- Split data into train and test sets to eliminate data leakage.

### 3. Model Training & Comparison
Benchmarked parametric, non-parametric, and tree-based classification models:
- **Logistic Regression** (Linear baseline)
- **K-Nearest Neighbors (KNN)** (Distance-based instance classifier)
- **Gaussian Naive Bayes** (Probabilistic baseline)
- **Support Vector Classifier (SVC)** (Hyperplane margin maximization)
- **Decision Tree Classifier** (Interpretable visual decision rules via `plot_tree`)
- **Random Forest Classifier** (Ensemble bagging to reduce variance)

### 4. Evaluation Metrics
- Evaluated models using **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Confusion Matrix**, and **ROC-AUC curves**.
- Serialized optimal artifacts (`best_model.pkl` and `scaler.pkl`) with `pickle`.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:**
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn (`train_test_split`, `StandardScaler`, `plot_tree`, metrics)
  - Model Export: Pickle
- **Environment:** Jupyter Notebook / Anaconda

---

## 📁 Repository Structure

```text
├── data/
│   └── student_dataset_10000_rows.csv     # Student records dataset
├── notebooks/
│   └── student_dataset.ipynb              # EDA, preprocessing & ML pipeline
├── models/
│   ├── best_model.pkl                     # Serialized classifier
│   └── scaler.pkl                         # Fitted StandardScaler object
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation
```

### 1. Clone the Repository
```Bash
git clone url?id=8.git
cd Student-Placement-Prediction-
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook notebooks/student_dataset.ipynb
```

### 📈 Key Insights & Takeaways
-Attendance percentage, assignment completion rate, and semester exam scores serve as the strongest predictors for placement success.

-Decision Tree visualization provides an interpretable decision path for academic counseling.

---

## 👤 Author
- **Aniket Yadav**
- GitHub: [@Yadavaniket19](https://github.com/Yadavaniket19)

---
⭐️ If you found this project helpful, please consider giving it a star!

