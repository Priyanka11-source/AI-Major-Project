# Cardiovascular Disease (CVD) General Health Prediction
### **AI Internship Major Project | Elewayte**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scipy-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-v2.2-green.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning research and exploratory data analysis project focused on predicting patient general health status based on clinical history, lifestyle behaviors, and biological demographics. Using a cleaned CDC dataset of **308,854 records**, this repository implements a complete end-to-end predictive pipeline and features a simulated production AI diagnostic assistant.

---

## 📋 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Dataset Description & Descriptive Statistics](#-dataset-description--descriptive-statistics)
   - [Column Schema](#column-schema)
   - [Descriptive Statistics for Numerical Features](#descriptive-statistics-for-numerical-features)
   - [Target Class Distribution & Imbalance](#target-class-distribution--imbalance)
3. [Machine Learning Pipeline](#-machine-learning-pipeline)
   - [1. Data Preprocessing](#1-data-preprocessing)
   - [2. Model Implementations & Mathematical Architectures](#2-model-implementations--mathematical-architectures)
   - [3. SVM Computational Complexity Analysis](#3-svm-computational-complexity-analysis)
4. [Model Performance & Evaluation Metrics](#-model-performance--evaluation-metrics)
   - [Accuracy Comparison](#accuracy-comparison)
   - [Detailed Classification Reports](#detailed-classification-reports)
   - [Confusion Matrices Analysis](#confusion-matrices-analysis)
5. [Hyperparameter Tuning & Optimization](#-hyperparameter-tuning--optimization)
6. [AI Health Prediction System Simulation](#-ai-health-prediction-system-simulation)
7. [Project Structure](#-project-structure)
8. [Installation & Local Execution](#-installation--local-execution)
9. [Future Scope & Key Takeaways](#-future-scope--key-takeaways)

---

## 🔍 Project Overview & Objectives

Predicting self-reported general health status (`Poor`, `Fair`, `Good`, `Very Good`, or `Excellent`) from objective clinical indicators and lifestyle habits is a complex multi-class classification challenge. Patient self-assessment is highly subjective, yet it serves as a powerful proxy for underlying chronic conditions.

The primary objectives of this project are:
1. **Explore and Standardize** the Cardiovascular Disease dataset (`CVD_cleaned.csv`).
2. **Build and Train** three distinct classifier families: **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)** using an 80:20 train-test split.
3. **Analyze Performance Metrics** using classification reports, confusion matrices, and accuracy comparison visualizations.
4. **Optimize Model Hyperparameters** via randomized grid search (`RandomizedSearchCV`) on the top-performing ensemble model.
5. **Develop a Production-Style Simulator** that ingests a raw patient profile, runs preprocessing on-the-fly, executes model inference, and compares the AI's diagnosis against the actual clinical record.

---

## 📊 Dataset Description & Descriptive Statistics

The project leverages the [CVD_cleaned.csv](file:///c:/Users/priya/OneDrive/Desktop/Skills/Elewayte/AI%20Major%20Project/CVD_cleaned.csv) dataset containing **308,854 rows** and **19 columns** with zero missing values.

### Column Schema

| Column Name | Data Type | Variable Type | Description |
| :--- | :--- | :--- | :--- |
| **General_Health** | Categorical | Target Variable | Self-reported health status (`Poor`, `Fair`, `Good`, `Very Good`, `Excellent`) |
| **Checkup** | Categorical | Feature | Time since last routine clinical checkup |
| **Exercise** | Categorical | Feature | Physical exercise in the past 30 days (`Yes`/`No`) |
| **Heart_Disease** | Categorical | Feature | Diagnosed with heart disease (`Yes`/`No`) |
| **Skin_Cancer** | Categorical | Feature | Diagnosed with skin cancer (`Yes`/`No`) |
| **Other_Cancer** | Categorical | Feature | Diagnosed with other forms of cancer (`Yes`/`No`) |
| **Depression** | Categorical | Feature | Diagnosed with depression/depressive disorder (`Yes`/`No`) |
| **Diabetes** | Categorical | Feature | Diagnosed diabetes status (`Yes`, `No`, pre-diabetes, etc.) |
| **Arthritis** | Categorical | Feature | Diagnosed with arthritis (`Yes`/`No`) |
| **Sex** | Categorical | Feature | Biological sex (`Male`/`Female`) |
| **Age_Category** | Categorical | Feature | Grouped age ranges (e.g., `18-24`, `55-59`, `80+`) |
| **Height_(cm)** | Numeric | Feature | Patient height in centimeters |
| **Weight_(kg)** | Numeric | Feature | Patient weight in kilograms |
| **BMI** | Numeric | Feature | Body Mass Index (calculated from height & weight) |
| **Smoking_History**| Categorical | Feature | Smoked 100+ cigarettes in lifetime (`Yes`/`No`) |
| **Alcohol_Consumption** | Numeric | Feature | Number of alcoholic drinks consumed per week |
| **Fruit_Consumption** | Numeric | Feature | Frequency of fruit consumption per day |
| **Green_Vegetables_Consumption** | Numeric | Feature | Frequency of green vegetable consumption per day |
| **FriedPotato_Consumption** | Numeric | Feature | Frequency of fried potato/french fry consumption per day |

### Descriptive Statistics for Numerical Features

Below are the computed statistical parameters for the numerical attributes in the dataset:

| Feature Name | Mean | Std Dev | Min | 25% | 50% | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Height (cm)** | 170.62 | 10.66 | 91.00 | 163.00 | 170.00 | 178.00 | 241.00 |
| **Weight (kg)** | 83.59 | 21.34 | 24.95 | 68.04 | 81.65 | 95.25 | 293.02 |
| **BMI** | 28.63 | 6.52 | 12.02 | 24.21 | 27.44 | 31.85 | 99.33 |
| **Alcohol Consumption** | 5.10 | 8.20 | 0.00 | 0.00 | 1.00 | 6.00 | 30.00 |
| **Fruit Consumption** | 29.84 | 24.88 | 0.00 | 12.00 | 30.00 | 30.00 | 120.00 |
| **Green Vegetables** | 15.11 | 14.93 | 0.00 | 4.00 | 12.00 | 20.00 | 128.00 |
| **Fried Potato** | 6.30 | 8.58 | 0.00 | 2.00 | 4.00 | 8.00 | 128.00 |

### Target Class Distribution & Imbalance

The target variable `General_Health` exhibits significant class imbalance:
- **Very Good**: 110,395 (35.7%)
- **Good**: 95,364 (30.9%)
- **Excellent**: 55,954 (18.1%)
- **Fair**: 35,810 (11.6%)
- **Poor**: 11,331 (3.7%)

*Implication:* The model is heavily exposed to majority classes (`Very Good`, `Good`), making it highly challenging to predict minority classes (`Poor`, `Fair`) without advanced resampling techniques.

---

## 📈 Machine Learning Pipeline

The pipeline is implemented cell-by-cell in the [Major.ipynb](file:///c:/Users/priya/OneDrive/Desktop/Skills/Elewayte/AI%20Major%20Project/Major.ipynb) Jupyter Notebook.

### 1. Data Preprocessing
*   **Categorical Encoding:** `LabelEncoder` was applied to convert text columns to ordinal integers. Target class encoding maps alphabetically:
    - `Excellent` $\rightarrow$ `0`
    - `Fair` $\rightarrow$ `1`
    - `Good` $\rightarrow$ `2`
    - `Poor` $\rightarrow$ `3`
    - `Very Good` $\rightarrow$ `4`
*   **Feature Scaling:** Standard scaling is crucial for models relying on distance metrics or gradient descent (SVM and Logistic Regression). Features were scaled using:
    $$z = \frac{x - \mu}{\sigma}$$
    where $\mu$ is the feature mean and $\sigma$ is the standard deviation.
*   **Train-Test Split:** Splitting is performed with an 80:20 ratio using a fixed `random_state=42` to guarantee reproducibility.
    - **Training Set:** 247,083 samples
    - **Testing Set:** 61,771 samples

### 2. Model Implementations & Mathematical Architectures
*   **Logistic Regression:**
    Uses a softmax activation function for multi-class classification, minimizing cross-entropy loss.
    - Configured with `max_iter=1000` to ensure solver convergence.
*   **Random Forest Classifier:**
    An ensemble bagging algorithm constructing multiple decision trees. The final output is based on majority voting. Captures complex feature interactions (like high BMI combined with depression and smoking) without assuming linearity.
*   **Support Vector Machine (SVM):**
    Projects the features into a higher-dimensional space to locate an optimal separating hyperplane. Uses the **Radial Basis Function (RBF) Kernel**:
    $$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

### 3. SVM Computational Complexity Analysis

Training a standard kernelized SVM requires solving a quadratic programming problem. The time complexity scales between:
$$\mathcal{O}(N^2) \quad \text{and} \quad \mathcal{O}(N^3)$$
where $N$ is the number of training instances.

- **For the full training set ($N = 247,083$):** An $\mathcal{O}(N^2)$ operation represents over $6.1 \times 10^{10}$ computations, taking hours to train on typical consumer CPUs.
- **Verification Strategy:** To circumvent this, the model was validated using a downsampled subset of **10,000 training samples**. This sub-train finished execution in **~8.12 seconds** while preserving boundary generalizability.

---

## 📋 Model Performance & Evaluation Metrics

### Accuracy Comparison

| Model | Training Dataset Size | Test Accuracy Score |
| :--- | :---: | :---: |
| **Logistic Regression (Baseline)** | 247,083 (Full) | **42.47%** |
| **Support Vector Machine (Baseline)** | 10,000 (Subsampled) | **41.93%** |
| **Random Forest (Baseline)** | 247,083 (Full) | **40.67%** |
| **Random Forest (Tuned)** | 20,000 (Subsampled Grid) | **41.62%** |

### Detailed Classification Reports

#### Logistic Regression (Accuracy: 42.47%)
```
              precision    recall  f1-score   support

           0       0.46      0.13      0.20     11339
           1       0.37      0.13      0.19      7166
           2       0.41      0.44      0.42     18784
           3       0.40      0.05      0.09      2265
           4       0.44      0.70      0.54     22217

    accuracy                           0.42     61771
   macro avg       0.41      0.29      0.29     61771
weighted avg       0.42      0.42      0.38     61771
```

#### Random Forest Classifier (Accuracy: 40.67%)
```
              precision    recall  f1-score   support

           0       0.40      0.24      0.30     11339
           1       0.33      0.18      0.23      7166
           2       0.39      0.46      0.42     18784
           3       0.32      0.08      0.13      2265
           4       0.44      0.55      0.49     22217

    accuracy                           0.41     61771
   macro avg       0.37      0.30      0.31     61771
weighted avg       0.40      0.41      0.39     61771
```

### Confusion Matrices Analysis

*   **Logistic Regression:**
    ```
    Actual \ Pred    Excellent [0]   Fair [1]   Good [2]   Poor [3]   Very Good [4]
    Excellent [0]        1452           30        1401          1            8455
    Fair [1]               63          906        4144         99            1954
    Good [2]              468          690        8228         62            9336
    Poor [3]               10          569        1243        117             326
    Very Good [4]        1176          226        5272         14           15529
    ```
    *Insight:* Logistic Regression heavily predicts the majority class `Very Good` [4] (yielding 15,529 correct predictions but misclassifying 8,455 `Excellent` patients as `Very Good`). It struggles with `Poor` [3], obtaining a low recall of only 5% (117 / 2,265).

*   **Random Forest Classifier:**
    ```
    Actual \ Pred    Excellent [0]   Fair [1]   Good [2]   Poor [3]   Very Good [4]
    Excellent [0]        2741          108        2159          7            6324
    Fair [1]              184         1279        3832        213            1658
    Good [2]             1168         1249        8635        134            7598
    Poor [3]               34          696        1047        182             306
    Very Good [4]        2745          536        6615         38           12283
    ```
    *Insight:* Random Forest produces more balanced predictions across minority classes. For instance, it correctly predicts 182 `Poor` [3] profiles (8% recall) and 2,741 `Excellent` [0] profiles (24% recall), reflecting better sensitivity to non-linear lifestyle thresholds than Logistic Regression.

---

## 🛠️ Hyperparameter Tuning & Optimization

To improve the Random Forest model and prevent overfitting, we perform a hyperparameter grid search using `RandomizedSearchCV` with 3-fold cross-validation (`cv=3`):

```python
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### Tuning Results
Following 5 search iterations over a 20,000-sample training dataset, the optimal parameters discovered were:
- **`n_estimators`**: `200` (Dense tree ensemble)
- **`max_depth`**: `20` (Constraining deep noise fitting)
- **`min_samples_split`**: `5`
- **`min_samples_leaf`**: `4` (Restricts individual leaf size, reducing variance)

**Performance Lift:** Tuning raised the test accuracy of the downsampled Random Forest to **41.62%** (outperforming the baseline Random Forest by **0.95%**), demonstrating the efficacy of restricting individual tree depth and increasing ensemble size.

---

## 💻 AI Health Prediction System Simulation

An end-to-end simulated inference function (`ai_health_prediction`) was constructed to represent how the model operates in a clinical software application:

```text
               [Patient Sample Index]
                         │
                         ▼
             [Extract Raw Profile Row]
                         │
                         ▼
           [Decode Categoricals to Text]  ──► (Console Display Profile)
                         │
                         ▼
             [Numeric Feature Scaling]    ──► (Applies StandardScaler)
                         │
                         ▼
             [Trained Model Inference]    ──► (Executes Best Estimator)
                         │
                         ▼
             [Decoder Output Mapping]     ──► (Labels: EXCELLENT, POOR, etc.)
                         │
                         ▼
       [Diagnostic Comparison vs. Actual]
```

### Execution Example Trace:
```text
**************************************************
      AI HEALTH PREDICTION SYSTEM      
**************************************************

[Patient Profile Data Received]
 - Checkup: Within the past year
 - Exercise: Yes
 - Heart_Disease: No
 - Skin_Cancer: No
 - Other_Cancer: No
 - Depression: No
 - Diabetes: No
 - Arthritis: No
 - Sex: Female
 - Age_Category: 80+
 - Height_(cm): 152.0
 - Weight_(kg): 52.16
 - BMI: 22.44
 - Smoking_History: No
 - Alcohol_Consumption: 0.0
 - Fruit_Consumption: 30.0
 - Green_Vegetables_Consumption: 30.0
 - FriedPotato_Consumption: 0.0
--------------------------------------------------
>> PREDICTED General Health Status : VERY GOOD
>> ACTUAL General Health Record    : VERY GOOD
**************************************************
```

---

## 📁 Project Structure

```text
AI-Major-Project/
│
├── CVD_cleaned.csv               # The cleaned CDC Cardiovascular Disease dataset (29.3 MB)
├── General health prediction.docx  # Grading rubric & project criteria documentation
├── Major.ipynb                   # The primary pipeline Notebook (Data, Models, Search & Sim)
├── README.md                     # Comprehensive project documentation (this file)
└── .gitignore                    # Local environment and Jupyter checkpoint exclusions
```

---

## 🚀 Installation & Local Execution

### 1. Prerequisites & Environment Setup
Clone the repository and install the verified dependency stack:

```bash
# Clone the repository
git clone https://github.com/Priyanka11-source/AI-Major-Project.git
cd "AI Major Project"

# Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Running the Pipeline
Open the notebook in Jupyter or VS Code:
```bash
jupyter notebook Major.ipynb
```
Select the virtual environment kernel and run the notebook cells sequentially to execute data cleaning, baseline training, hyperparameter searches, and the interactive simulator.

---

## 💡 Future Scope & Key Takeaways

1. **Subjective Health Scoring:** The target `General_Health` metric is self-reported, leading to overlap between neighboring classes (e.g., `Good` vs. `Very Good`). This bounds deterministic classification accuracy around ~42-45%.
2. **Addressing Class Imbalance:** Future iterations will implement **SMOTE (Synthetic Minority Over-sampling Technique)** or utilize cost-sensitive class weights (`class_weight='balanced'`) to boost recall on the critical `Poor` health class.
3. **Alternative Kernels for Scale:** To leverage Support Vector machines on large clinical datasets, linear approximations or approximation kernels (e.g., `Nystroem` or `SGDClassifier(loss='hinge')`) should be utilized instead of exact RBF kernels to reduce mathematical training time to $\mathcal{O}(N)$.
