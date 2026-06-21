# Cardiovascular Disease (CVD) General Health Prediction - AI Major Project

**AI Internship Major Project | Elewayte**

A comprehensive machine learning and exploratory data analysis project focused on predicting patient general health status based on clinical history, lifestyle behaviors, and biological demographics using a cleaned dataset of 308,854 records. This repository represents the final Major Project submission for the AI Internship program at **Elewayte**.

---

## 📋 Table of Contents
- [Project Overview & Objectives](#-project-overview--objectives)
- [Dataset Characteristics](#-dataset-characteristics)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Implementations & Results](#2-model-implementations--results)
  - [3. Accuracy Comparison](#3-accuracy-comparison)
  - [4. Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [5. AI Health Prediction System Simulation](#5-ai-health-prediction-system-simulation)
- [Project Structure](#-project-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [Technology Stack](#-technology-stack)
- [Summary of Key Insights](#-summary-of-key-insights)

---

## 🔍 Project Overview & Objectives

The primary objective of this project is to develop and evaluate predictive models that assess a patient's self-reported general health status (`Poor`, `Fair`, `Good`, `Very Good`, or `Excellent`). By analyzing clinical history (e.g., heart disease, diabetes, depression) alongside lifestyle habits (e.g., exercise, alcohol, fruit/vegetable/potato consumption) and biological features (height, weight, BMI), the project simulates an AI diagnostic assistant.

Key requirements addressed in the project:
1. Load and clean the Cardiovascular Disease dataset (`CVD_cleaned.csv`).
2. Train-test split with an 80:20 ratio.
3. Train and compare three classifiers: **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)**.
4. Output classification reports, accuracy scores, and confusion matrices for each model.
5. Create a visual comparison of model accuracies.
6. Apply randomized grid search (`RandomizedSearchCV`) to optimize model hyperparameters.
7. Build an interactive prediction function that takes a patient profile, performs preprocessing, and outputs the simulated AI health diagnostic prediction compared to the actual patient record.

---

## 📊 Dataset Characteristics

The project utilizes the [CVD_cleaned.csv](file:///c:/Users/priya/OneDrive/Desktop/Skills/Elewayte/AI%20Major%20Project/CVD_cleaned.csv) dataset containing **308,854 rows** and **19 columns** with the following details:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **General_Health** | Categorical | Target variable. Self-reported health status (`Poor`, `Fair`, `Good`, `Very Good`, `Excellent`) |
| **Checkup** | Categorical | Time since the last routine clinical checkup |
| **Exercise** | Categorical | Engaged in physical exercise in the past 30 days (`Yes`/`No`) |
| **Heart_Disease** | Categorical | Diagnosed with heart disease (`Yes`/`No`) |
| **Skin_Cancer** | Categorical | Diagnosed with skin cancer (`Yes`/`No`) |
| **Other_Cancer** | Categorical | Diagnosed with any other form of cancer (`Yes`/`No`) |
| **Depression** | Categorical | Diagnosed with depression or depressive disorder (`Yes`/`No`) |
| **Diabetes** | Categorical | Diagnosed diabetes status (e.g., `Yes`, `No`, `No, pre-diabetes`, or `Yes, during pregnancy`) |
| **Arthritis** | Categorical | Diagnosed with arthritis (`Yes`/`No`) |
| **Sex** | Categorical | Biological sex (`Male`/`Female`) |
| **Age_Category** | Categorical | Grouped age ranges (e.g., `18-24`, `55-59`, `80+`) |
| **Height_(cm)** | Numeric | Patient height in centimeters |
| **Weight_(kg)** | Numeric | Patient weight in kilograms |
| **BMI** | Numeric | Body Mass Index calculated from height and weight |
| **Smoking_History**| Categorical | Patient smoking history (`Yes`/`No` representing if they smoked 100+ cigarettes) |
| **Alcohol_Consumption** | Numeric | Number of alcoholic drinks consumed per week |
| **Fruit_Consumption** | Numeric | Frequency of fruit consumption per day |
| **Green_Vegetables_Consumption** | Numeric | Frequency of green vegetable consumption per day |
| **FriedPotato_Consumption** | Numeric | Frequency of fried potato/french fry consumption per day |

---

## 📈 Machine Learning Pipeline

The entire machine learning workflow is implemented and detailed inside the [Major.ipynb](file:///c:/Users/priya/OneDrive/Desktop/Skills/Elewayte/AI%20Major%20Project/Major.ipynb) notebook.

### 1. Data Preprocessing
- **Categorical Encoding:** `LabelEncoder` was applied to encode all object/string variables (such as `Sex`, `Exercise`, `Diabetes`, and the target variable `General_Health`) into unique integers.
- **Feature Scaling:** Since clinical models are sensitive to varying scales (especially Logistic Regression and SVM), `StandardScaler` was used to standardize numeric attributes to zero mean and unit variance.
- **Train-Test Split:** The preprocessed dataset was split into **80% training data (247,083 samples)** and **20% testing data (61,771 samples)**.

### 2. Model Implementations & Results
Each classifier was trained on the training set and evaluated on the test set using standard metrics:

*   **Logistic Regression:**
    - Configured with `max_iter=1000` to ensure convergence.
    - **Accuracy:** **42.47%**
    - High recall for the majority classes but lower sensitivity to minor classes.

*   **Random Forest Classifier:**
    - Trained using standard decision tree ensembles (`n_jobs=-1` for parallel CPU execution).
    - **Accuracy:** **40.67%**
    - Captured non-linear interactions between lifestyle choices and health outcomes.

*   **Support Vector Machine (SVM):**
    - Configured with the default RBF kernel.
    - **Training Speed Note:** Running a full SVM with an RBF kernel has $O(N^2)$ to $O(N^3)$ computational complexity. On the full dataset of 247,000+ samples, training takes hours. To verify, a downsampled subset of **10,000 training samples** was trained in **~8.12 seconds**, achieving an accuracy of **41.10%** on the test set.

### 3. Accuracy Comparison
A Matplotlib bar chart plots the three baseline accuracies side-by-side:
- **Logistic Regression:** 0.4247
- **Support Vector Machine (10k sample baseline):** 0.4110
- **Random Forest:** 0.4067

### 4. Hyperparameter Tuning
Using `RandomizedSearchCV` on the Random Forest classifier, we tune parameters across 3-fold cross-validation (`cv=3`):
- `n_estimators`: [50, 100, 200]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
This allows optimization of the decision boundaries and prevents overfitting.

### 5. AI Health Prediction System Simulation
A custom Python simulation function (`ai_health_prediction`) was written to emulate a production environment:
1. It takes a raw row index from the dataset.
2. It prints out the patient's lifestyle and clinical metrics in a readable dictionary format (decoding categorical integers back to strings).
3. It scales the patient's record on-the-fly using the pre-fitted `StandardScaler`.
4. It feeds the scaled metrics to the trained model and decodes the numeric prediction back into a text label.
5. It prints the predicted health status alongside the actual clinical record.

---

## 📁 Project Structure

```
AI Major Project/
│
├── CVD_cleaned.csv             # The cleaned CDC Cardiovascular Disease dataset (29.3 MB)
├── General health prediction.docx # Project specifications and grading rubric
├── Major.ipynb                 # The complete Jupyter notebook with models and simulation
├── README.md                   # Full project documentation (this file)
└── .gitignore                  # Excludes Jupyter checkpoints and pycache from commits
```

---

## 🚀 Getting Started & Installation

### Prerequisites
Make sure you have a working Python 3.8+ installation and install the required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the Notebook
1. Clone or navigate to the directory:
   ```bash
   cd "AI Major Project"
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook Major.ipynb
   ```
3. Run the cells sequentially to load data, train the baseline models, run the Randomized Search hyperparameter tuning, and test the AI diagnosis simulation.

---

## 🛠️ Technology Stack
- **Language:** Python 3.13
- **Data Wrangling:** Pandas & NumPy
- **Model Training & Evaluation:** Scikit-Learn (LogisticRegression, RandomForestClassifier, SVC, RandomizedSearchCV, StandardScaler)
- **Data Visualization:** Matplotlib & Seaborn
- **Development Environment:** Jupyter Notebook (Colab Compatible)

---

## 💡 Summary of Key Insights
1. **Multi-class Classification Challenge:** Predicting 5 distinct levels of general health based on self-reported risk factors is a highly complex problem. General health is subjective, which contributes to the baseline accuracies hovering around 40-42%.
2. **Computational Constraints of SVM:** While RBF SVM provides high boundary accuracy on small sets, training standard kernelized SVM on large datasets (300k+ rows) becomes computationally prohibitive, highlighting the superiority of Random Forest and Logistic Regression for scalable health applications.
3. **Hyperparameter Improvements:** Decision trees benefit strongly from minimum sample splits and leaf limits to prevent individual trees from learning dataset noise.
