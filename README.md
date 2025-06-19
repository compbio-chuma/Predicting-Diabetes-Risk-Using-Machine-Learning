# Diabetes Risk Prediction using Machine Learning

A beginner-friendly AI in Healthcare project that predicts the likelihood of diabetes using patient health data. This project walks through the entire machine learning pipeline, from data preprocessing to model evaluation and basic deployment with Streamlit.

---

##  Project Overview

**Goal:**  
Develop a supervised machine learning model that can **predict the onset of diabetes** based on diagnostic measurements and patient health records.

This project was built for learning and research purposes to gain practical familiarity with:
- Machine Learning in Healthcare
- Model interpretability
- Streamlit-based deployment
- Responsible AI and healthcare ethics

---

##  Problem Statement

Diabetes is a chronic disease affecting millions worldwide. Early prediction and diagnosis are critical to preventive care. This project explores whether we can train a machine learning model using medical features (such as glucose level, BMI, insulin, age) to **predict diabetes risk**.

**Machine Learning Task:** Binary classification  
**Target variable:** `Outcome`  
- 0: No diabetes  
- 1: Diabetes diagnosed

---

## 📊 Dataset

- **Source:** [PIMA Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples:** 768 female patients (21 years and older)
- **Features (columns):**
  - `Pregnancies`
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `Insulin`
  - `BMI`
  - `DiabetesPedigreeFunction`
  - `Age`
  - `Outcome` (target)

**Note:** The dataset includes some invalid 0 values for features like BMI, Glucose, etc. These were handled in preprocessing.

---

## 🛠️ Tools & Technologies

| Tool / Library      | Purpose                                 |
|---------------------|------------------------------------------|
| **Python**          | Core programming language                |
| **Jupyter Notebook**| Exploratory data analysis & prototyping |
| **pandas**          | Data manipulation                        |
| **scikit-learn**    | ML models, preprocessing, metrics        |
| **matplotlib / seaborn** | Data visualization                 |
| **joblib**          | Model serialization                      |
| **Streamlit**       | Web app deployment                       |
| **Git & GitHub**    | Version control and project hosting      |

---

##  Machine Learning Models

We trained and evaluated the following models:
- **Logistic Regression**: Baseline linear classifier
- **Decision Tree**: Non-linear classifier, interpretable
- **Random Forest**: Ensemble method for higher accuracy

Each model was evaluated using:
- Accuracy
- Precision & Recall
- F1 Score
- ROC AUC
- Confusion Matrix

---

##  Project Pipeline

### 1. 📂 Data Ingestion
- Load raw CSV data into pandas
- Handle invalid 0 values by replacing with NaNs

### 2.  Data Cleaning & Preprocessing
- Median imputation for missing values
- Standard scaling of numeric features

### 3. 📊 Exploratory Data Analysis (EDA)
- Visualize class imbalance
- Check feature distributions
- Compute correlation matrix

### 4. 🧠 Model Training
- Train/test split (80/20)
- Train Logistic Regression, Decision Tree, and Random Forest
- Use joblib to save models

### 5. 📏 Model Evaluation
- Evaluate all models on the test set
- Visualize performance metrics and ROC curves

### 6. 🌐 Deployment with Streamlit
- Build an interactive web app that allows users to input features and predict diabetes risk
- Hosted locally using Streamlit

---

## 💻 Project Structure
diabetes-risk-prediction/
├── data/
│   ├── raw/             # Original downloaded dataset
│   └── processed/       # Cleaned and imputed dataset
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Data cleaning and preprocessing
│   ├── 02_eda.ipynb                # Exploratory Data Analysis (EDA)
│   ├── 03_model_training.ipynb     # Model training and serialization
│   └── 04_model_evaluation.ipynb   # Model testing and evaluation
├── src/
│   └── app.py           # Streamlit web application for user interaction
├── outputs/
│   ├── figures/         # Generated plots and visualizations
│   └── metrics/         # Saved models and scalers (e.g., .pkl files)
├── requirements.txt     # Python dependencies
├── README.md            # Project overview and documentation
├── LICENSE              # Open-source license (e.g., MIT)
└── .gitignore           # Files/folders to ignore in version control


