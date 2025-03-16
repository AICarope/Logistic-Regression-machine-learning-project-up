# Heart Attack Risk Prediction in Women  

## Introduction  

Cardiovascular disease (CVD) is the leading cause of death among women globally, responsible for **30% of female deaths** each year. Women often experience **severe and atypical symptoms** like **fatigue, nausea, and jaw pain**, which traditional diagnostic models often overlook.  

Using a dataset from Kaggle, containing **8,763 patient records** (including **2,652 female patients**), and leveraging **Python, Logistic Regression, and Machine Learning** techniques, this project aims to predict **heart attack risk in women** and identify key contributing factors. The model analyzes **health, lifestyle, and demographic variables** to enhance **early detection, prevention, and healthcare interventions**.  

---

## 📂 Materials  

📌 **Access the analysis notebooks:**  
- [📊 Exploratory Data Analysis (EDA)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/2_Women_EDA.ipynb)  
- [🤖 Machine Learning Models (ML)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/3_Women_ML.ipynb)  

---

## 🔬 Methodology  

### **Dataset Overview**  

The dataset captures **key attributes** critical for understanding cardiovascular health in women:  

- **Demographics:** Age, Sex, Income, Country, Continent, Hemisphere  
- **Health Metrics:** Cholesterol, Blood Pressure, Heart Rate, Diabetes, BMI, Triglycerides  
- **Lifestyle Factors:** Smoking, Alcohol Consumption, Exercise Hours, Diet, Physical Activity, Sedentary Hours, Sleep Hours  
- **Medical History:** Family History of Heart Disease, Previous Heart Problems, Medication Use, Stress Level  

📌 **Dataset Name:** `female_heart_df2`  
📌 **Source:** Kaggle - [Heart Attack Prediction Dataset](https://github.com/user-attachments/assets/4eba3d96-36ef-4a7c-8df7-a1ab4b0ab19f)  

### **Target and Predictor Variables**  

- **X (Predictors):** Health Metrics, Lifestyle Factors, Medical History  
- **Y (Target):** Female Heart Attack Risk (**1 = At-risk, 0 = No-risk**)  
  - **1,708 no-risk cases**  
  - **944 at-risk cases**  

📊 **Data Distribution:**  
![Heart Attack Risk Data](https://github.com/user-attachments/assets/a536a4c3-cb06-4b77-87d9-2ef562f58a86)  

---

## 🛠 Data Cleaning & Preprocessing  

To ensure data integrity, the following preprocessing steps were applied:  

1. **Handling Missing Values:**  
   - Numerical values were **imputed with mean/median**.  
   - Categorical values were **encoded into numerical representations**.  

2. **Feature Scaling:**  
   - Standardization was applied to numerical features to **ensure uniform scaling**.  

3. **Removing Duplicates & Outliers:**  
   - Duplicate entries were removed.  
   - **Interquartile Range (IQR) & Z-score techniques** were used for outlier detection.  

📊 **Preprocessed Data Overview:**  
![Data Cleaning](https://github.com/user-attachments/assets/0816a471-39f5-429f-8b1d-956723f71b15)  

---

## 📈 Exploratory Data Analysis (EDA)  

### **Geographic Trends in Smoking**  

- **Highest % of Female Smokers by Continent:**  
  - 🌍 **Africa:** **68.8%**  
  - 🌏 **Europe:** **67.7%**  
  - 🌎 **Asia/South America:** **65.7%**  

- **Top 3 Countries with Most Female Smokers:**  
  - 🇮🇹 **Italy:** **72.1%**  
  - 🇿🇦 **South Africa:** **70.9%**  
  - 🇻🇳 **Vietnam:** **70.3%**  

📊 **Smoking Correlation Analysis:**  
![Smoking Correlation](https://github.com/user-attachments/assets/bdcb19f3-82a0-4a67-b3cb-13353dd3b340)  

### **Smoking & Heart Attack Risk**  

- **Survival Rates Without Heart Attack Risk:**  
  - Smokers aged **40-59**: **64.83%**  
  - Smokers over **59**: **64.81%**  

- **Survival Rates With Heart Attack Risk:**  
  - Smokers aged **40-59**: **35.17%**  
  - Smokers over **59**: **35.19%**  

📊 **Survival Trends in Smokers:**  
![Survival Rates](https://github.com/user-attachments/assets/f4b3cdb0-1c03-439b-9668-ca8451df9914)  

---

## 🤖 Machine Learning Models  

Baseline Model:  
📊 **Baseline Performance Overview:**  
![Baseline Model](https://github.com/user-attachments/assets/0fdce8aa-75c3-45a6-939c-93b87d89fac5)  

### **Performance Comparison of ML Models**  

| **Model** | **Accuracy** | **Recall (High-Risk)** | **F1-Score (High-Risk)** | **Notes** |
|-----------|------------|------------------|------------------|----------------------------|
| **Baseline Logistic Regression (PCA/No PCA)** | **0.6667** | **0.01** | **0.02** | Biased toward Class 0, poor recall |
| **Random Forest (PCA)** | **0.6478** | **0.07** | -- | Slight recall improvement |
| **Decision Tree** | **0.6535** | **0.06** | -- | Marginal improvement |
| ✅ **Random Forest with SMOTE** | **0.6520** | **0.67** | **0.66** | **Best model; significantly improved recall** |
| **Best GridSearch Random Forest** | **0.6404** | **0.68** | -- | Best recall but lower accuracy |
| ❌ **XGBoost** | **0.5932** | **0.16** | -- | **Worst model; low accuracy and recall** |
| **Tuned Random Forest** | **0.6629** | **0.03** | -- | Very low recall |
| **Stacking Model** | **0.6681** | **0.03** | **0.05** | High accuracy but recall too low |

### **Key Findings**  

The **Random Forest with SMOTE model** performed the best, achieving:  
- **Accuracy:** **65.2%**  
- **Precision:** **67%**  
- **Recall:** **66%**  

SMOTE (Synthetic Minority Over-sampling Technique) effectively balanced the dataset, improving recall and ensuring fewer high-risk cases go undetected.  

📊 **Feature Importance for Heart Attack Prediction:**  
![Feature Importance](https://github.com/user-attachments/assets/8d02621c-b7f7-4398-b923-bc5e7396c383)  

---

## 📌 Conclusion  

This study identifies **key factors contributing to heart attack risk in women**, such as **stress, sleep patterns, sedentary behavior, and previous heart problems**. Addressing these through **lifestyle changes, stress management, and physical activity** can significantly reduce risk.  

**Future Research Directions:**  
- Incorporate **genetic data, diet, and medication use** to refine predictions.  
- **Expand the dataset** for better risk assessment.  
- Further **optimize machine learning models** with cost-sensitive learning techniques.  

This project demonstrates how **machine learning can improve heart attack risk prediction in women**, paving the way for **more personalized, AI-driven healthcare interventions**. 🚀  

---

This **Markdown file** is **fully formatted** with **consistent text sizes** and **properly displayed images** for **GitHub README.md**. 🎯 Copy & paste it directly into your repository! ✅🚀  
