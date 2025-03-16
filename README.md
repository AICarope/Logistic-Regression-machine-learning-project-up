# Heart Attack Risk Prediction in Women

## Introduction

Cardiovascular disease (CVD) is the leading cause of death among women globally, responsible for **30% of female deaths** each year. Women often experience **severe and atypical symptoms** like **fatigue, nausea, and jaw pain**, which traditional diagnostic models often overlook.

Using a dataset from Kaggle, containing **8,763 patient records** (including **2,652 female patients**), and leveraging **Python, Logistic Regression, and Machine Learning** techniques, this project aims to predict **heart attack risk in women** and identify key contributing factors. The model analyzes **health, lifestyle, and demographic variables** to enhance **early detection, prevention, and healthcare interventions**.

---

## 📂 Materials

📌 Access the analysis notebooks:

- 📊 **Exploratory Data Analysis (EDA)**[Python Notebook: Exploratory Data Analysis (EDA)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/2_Women_EDA.ipynb)![image](https://github.com/user-attachments/assets/af95277a-f593-403d-8d54-d4c189c8a5e9)
- 🤖 **Machine Learning Models (ML)**[Python Notebook: Machine Learning (ML)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/3_Women_ML.ipynb)![image](https://github.com/user-attachments/assets/48174e02-39a2-4f24-b2d8-360494dfa4b5)


---

## 🔬 Methodology

### Dataset Overview

The dataset captures **key attributes** critical for understanding cardiovascular health in women:

- **Demographics:** Age, Sex, Income, Country, Continent, Hemisphere
- **Health Metrics:** Cholesterol, Blood Pressure, Heart Rate, Diabetes, BMI, Triglycerides
- **Lifestyle Factors:** Smoking, Alcohol Consumption, Exercise Hours, Diet, Physical Activity, Sedentary Hours, Sleep Hours
- **Medical History:** Family History of Heart Disease, Previous Heart Problems, Medication Use, Stress Level

📌 **Dataset Name:** `female_heart_df2`  
📌 **Source:** Kaggle - *Heart Attack Prediction Dataset*

### Target and Predictor Variables

- **X (Predictors):** Health Metrics, Lifestyle Factors, Medical History  
- **Y (Target):** Female Heart Attack Risk (`1 = At-risk`, `0 = No-risk`)

- **1,708** no-risk cases
- **944** at-risk cases

---

## 🛠 Data Cleaning & Preprocessing

To ensure data integrity, the following preprocessing steps were applied:

### Handling Missing Values:
- Numerical values were **imputed with mean/median**.
- Categorical values were **encoded into numerical representations**.

### Feature Scaling:
- **Standardization** was applied to numerical features to **ensure uniform scaling**.

### Removing Duplicates & Outliers:
- **Duplicate entries** were removed.
- **Interquartile Range (IQR) & Z-score techniques** were used for outlier detection.

---

## 📈 Exploratory Data Analysis (EDA)

### Geographic Trends in Smoking

#### **Highest % of Female Smokers by Continent**:
- 🌍 **Africa:** 68.8%
- 🌍 **Europe:** 67.7%
- 🌍 **Asia/South America:** 65.7%

#### **Top 3 Countries with Most Female Smokers**:
1. 🇮🇹 **Italy:** 72.1%
2. 🇿🇦 **South Africa:** 70.9%
3. 🇻🇳 **Vietnam:** 70.3%

📊 **Smoking Correlation Analysis**:
- **Strongest positive correlation:** Smoking and Age (**0.81**) – *older women tend to smoke more*.

### Smoking & Heart Attack Risk

#### **Survival Rates Without Heart Attack Risk**:
- **Smokers aged 40-59:** 64.83%
- **Smokers over 59:** 64.81%

#### **Survival Rates With Heart Attack Risk**:
- **Smokers aged 40-59:** 35.17%
- **Smokers over 59:** 35.19%

### Other Variables:
- Many women in the dataset experience **moderate to high stress**, with levels **5 and 6 being most common**.
- Stress levels **9 and 10** are concerning, linked to **197 and 184 female smokers**.
- Smoking impacts **sleep**, with **69.4% getting only four hours** and **68.3% sleeping ten hours**.
- **Cholesterol increases with age**, affecting **456 young women (19-39), 399 adults (40-59), and 643 seniors (60+)**.
- **Heart rate analysis** shows most women have normal rates (**1,489**), but many experience **bradycardia (786) or tachycardia (377)**.
- **Diabetes significantly raises heart attack risk**, especially for **women aged 40-50**.

---

## 🤖 Machine Learning Models

### 📊 Baseline Performance Overview:

In evaluating the performance of various machine learning models, **accuracy, recall (sensitivity), and F1-score** were used to assess effectiveness.

| Model | Accuracy | Recall (High-Risk) | F1-Score (High-Risk) | Notes |
|-------|----------|--------------------|----------------------|--------|
| **Baseline Logistic Regression (PCA/No PCA)** | 0.6667 | 0.01 | 0.02 | Biased toward Class 0, poor recall |
| **Random Forest (PCA)** | 0.6478 | 0.07 | -- | Slight recall improvement |
| **Decision Tree** | 0.6535 | 0.06 | -- | Marginal improvement |
| ✅ **Random Forest with SMOTE** | 0.6520 | 0.67 | 0.66 | **Best model; significantly improved recall** |
| **Best GridSearch Random Forest** | 0.6404 | 0.68 | -- | Best recall but lower accuracy |
| ❌ **XGBoost** | 0.5932 | 0.16 | -- | Worst model; low accuracy and recall |
| **Tuned Random Forest** | 0.6629 | 0.03 | -- | Very low recall |
| **Stacking Model** | 0.6681 | 0.03 | 0.05 | High accuracy but recall too low |

### 🔹 **Key Findings**
The **Random Forest with SMOTE** model performed the best:

- **Accuracy:** 65.2%
- **Precision:** 67%
- **Recall:** 66%

**SMOTE (Synthetic Minority Over-sampling Technique)** effectively balanced the dataset, improving recall and ensuring fewer high-risk cases go undetected.

---

## 📊 Feature Importance for Heart Attack Prediction

Feature importance analysis showed that **stress level and previous heart problems** were the most significant predictors of heart attack risk.

| Feature | Importance (%) |
|---------|--------------|
| **Stress Level** | 7.33% |
| **Sleep Hours Per Day** | 6.18% |
| **Sedentary Hours Per Day** | 6.17% |
| **Previous Heart Problems** | 5.95% |
| **Alcohol Consumption** | 5.91% |
| **Cholesterol** | 5.88% |
| **Family History** | 5.87% |
| **Heart Rate** | 5.66% |
| **Obesity** | 5.56% |
| **Diabetes** | 5.51% |

---

## 📌 Conclusion

This study identifies **key factors contributing to heart attack risk in women**, such as **stress, sleep patterns, sedentary behavior, and previous heart problems**. Addressing these through **lifestyle changes, stress management, and physical activity** can significantly reduce risk.

### Future Research Directions:
- **Incorporate genetic data, diet, and medication use** to refine predictions.
- **Expand the dataset** for better risk assessment.
- **Further optimize machine learning models** with cost-sensitive learning techniques.

This project demonstrates how **machine learning can improve heart attack risk prediction in women**, paving the way for **more personalized, AI-driven healthcare interventions**. 🚀
