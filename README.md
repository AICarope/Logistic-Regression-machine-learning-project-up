# Logistic Regression of Heart Attack Risk Prediction in Women

## Introduction

Cardiovascular disease (CVD) is the leading cause of death among women globally, responsible for **30% of female deaths** each year. Women often experience **severe and atypical symptoms** like **fatigue, nausea, and jaw pain**, which traditional diagnostic models often overlook.

Using a dataset from Kaggle, containing **8,763 patient records** (including **2,652 female patients**), and leveraging **Python, Logistic Regression, and Machine Learning** techniques, this project aims to predict **heart attack risk in women** and identify key contributing factors. The model analyzes **health, lifestyle, and demographic variables** to enhance **early detection, prevention, and healthcare interventions**.

---
## Materials
You can access the materials by clicking

-[Python Notebook: Exploratory Data Analysis (EDA)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/2_Women_EDA.ipynb)

-[Python Notebook: Machine Learning (ML)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/3_Women_ML.ipynb)

---

## 🔬 Methodology

### Dataset Overview

The dataset captures **key attributes** critical for understanding cardiovascular health in women:

- **Demographics:** Age, Sex, Income, Country, Continent, Hemisphere
- **Health Metrics:** Cholesterol, Blood Pressure, Heart Rate, Diabetes, BMI, Triglycerides
- **Lifestyle Factors:** Smoking, Alcohol Consumption, Exercise Hours, Diet, Physical Activity, Sedentary Hours, Sleep Hours
- **Medical History:** Family History of Heart Disease, Previous Heart Problems, Medication Use, Stress Level
  
  - **Dataset Name:** `female_heart_df2`
  - **Source:** [Kaggle - Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)


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
- **Africa:** 68.8%
- **Europe:** 67.7%
- **Asia/South America:** 65.7%

#### **Top 3 Countries with Most Female Smokers**:
1. **Italy:** 72.1%
2. **South Africa:** 70.9%
3. **Vietnam:** 70.3%

📊 **Smoking Correlation Analysis**:
- **Strongest positive correlation:** Smoking and Age (**0.81**) – *older women tend to smoke more*.
  
![image](https://github.com/user-attachments/assets/5e54ffa6-d3c8-4f38-9b9c-e520c1841a4a)

### Smoking & Heart Attack Risk
Female smokers, particularly those over 59 years old (1,094 individuals), exhibit a high smoking prevalence, while 654 adult females aged 39-59 are also affected. Smoking significantly impacts heart attack risk and survival rates. Among smokers, survival without heart attack risk is nearly identical for adults (64.83%) and seniors (64.81%). However, when heart attack risk is present, survival drops to 35.17% for adults and 35.19% for seniors, highlighting smoking's severe cardiovascular effects. Additionally, 49.1% of female patients have a family history of heart-related problems. These findings emphasize the need for targeted interventions and prevention strategies for female smokers.

#### **Survival Rates Without Heart Attack Risk**:
- **Smokers aged 40-59:** 64.83%
- **Smokers over 59:** 64.81%

#### **Survival Rates With Heart Attack Risk**:
- **Smokers aged 40-59:** 35.17%
- **Smokers over 59:** 35.19%
![image](https://github.com/user-attachments/assets/05971317-0801-4aef-90e9-fecd9c7baa6f)

### Other Variables:
- Many women in the dataset experience **moderate to high stress**, with levels **5 and 6 being most common**.
- Stress levels **9 and 10** are concerning, linked to **197 and 184 female smokers**.
- Smoking impacts **sleep**, with **69.4% getting only four hours** and **68.3% sleeping ten hours**.
- **Cholesterol increases with age**, affecting **456 young women (19-39), 399 adults (40-59), and 643 seniors (60+)**.
- **Heart rate analysis** shows most women have normal rates (**1,489**), but many experience **bradycardia (786) or tachycardia (377)**.
- **Diabetes significantly raises heart attack risk**, especially for **women aged 40-50**.
![image](https://github.com/user-attachments/assets/2cdb6424-79d8-44ec-a34b-3ff4572650cf)

---

## 🤖 Machine Learning Models

### 📊 Baseline Performance Overview:

| Dataset Split  | X (Predictor Variables) | Y (Target Variable) | Proportion (%) |
|---------------|------------------------|---------------------|---------------|
| **Training Set** | 1856 | 1856 | 74.99 |
| **Testing Set**  | 619  | 619  | 25.01 |
| **Total**       | 2475 | 2475 | 100.0 |

In evaluating the performance of various machine learning models for predicting heart attack risk among female patients, several key metrics were utilized to assess their effectiveness. **Accuracy** was measured to determine the proportion of correct predictions made by each model. However, given the class imbalance in the dataset, with a higher number of 'no-risk' cases compared to 'at-risk' cases, additional metrics were crucial for a comprehensive evaluation. **Recall** (Sensitivity) for the 'at-risk' class was calculated to identify the model's ability to correctly detect actual positive cases, reflecting its effectiveness in identifying patients truly at risk. The **F1-Score** for the 'at-risk' class, which is the harmonic mean of precision and recall, was also computed to provide a balanced measure of the model's performance, especially in scenarios with imbalanced classes. These metrics collectively offered a nuanced understanding of each model's predictive capabilities, guiding the selection of the most appropriate algorithm for accurate heart attack risk prediction in the female cohort.

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

**SMOTE (Synthetic Minority Over-sampling Technique)** effectively balanced the dataset, improving recall and ensuring fewer high-risk cases go undetected. This means the model is better at correctly identifying positive heart attack cases, reducing the risk of missing high-risk individuals. Compared to other models, it provides a strong balance between precision and recall, making it a more reliable choice for heart attack risk prediction in women. The enhanced recall ensures that fewer high-risk cases go undetected, improving early intervention strategies.

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

In addition, for this analysis of heart attack risk among female patients, a notable collinearity was observed between the variables smoking and age. This relationship indicates that as age increases, the likelihood of smoking also rises, which can complicate the interpretation of each variable's individual impact on heart attack risk. Despite this collinearity, feature importance assessments from the machine learning models revealed that stress level and previous heart problems were more significant predictors of heart attack risk than smoking and age. This suggests that while smoking and age are related, their combined effect may be less critical in the presence of other dominant risk factors. Addressing collinearity is essential to ensure accurate model interpretations and to identify the most influential predictors for targeted.

---

## 📌 Conclusion

This study identifies **key factors contributing to heart attack risk in women**, such as **stress, sleep patterns, sedentary behavior, and previous heart problems**. Addressing these through **lifestyle changes, stress management, and physical activity** can significantly reduce risk.

### Future Research Directions:
- **Incorporate genetic data, diet, and medication use** to refine predictions.
- **Expand the dataset** for better risk assessment.
- **Further optimize machine learning models** with cost-sensitive learning techniques.

This project demonstrates how **machine learning can improve heart attack risk prediction in women**, paving the way for **more personalized, AI-driven healthcare interventions**. For instance this prototype [VivaHeart](https://github.com/AICarope/Deep-Learning-CNN-Artificial-Intelligence-Project) 🚀


