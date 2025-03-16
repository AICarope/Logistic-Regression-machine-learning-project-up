# Heart Attack Risk Prediction in Women

## Introduction  

Cardiovascular disease (CVD) is the leading cause of death among women globally, responsible for **30% of female deaths** each year. Women often experience more severe and atypical symptoms, such as **fatigue, nausea, and jaw pain**, which traditional diagnostic models frequently overlook.  

Using a dataset from Kaggle, containing 8,763 patient entries, with **2,652 female patients**, and by using Python, **Logistic Regression** and **Machine Learning** techniques, this project aims to predict heart attack risk in females and identify the most significant contributing factors. The model will analyze various **health, lifestyle, and demographic** variables to recommend **early detection, prevention, and healthcare interventions** for women at risk of heart disease. 

---

## Materials  

📂 **Access the analysis notebooks:**  
- [📊 Exploratory Data Analysis (EDA)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/2_Women_EDA.ipynb)  
- [🤖 Machine Learning Models (ML)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/3_Women_ML.ipynb)  

---

## Methodology  

### **Dataset Overview**  

The dataset captures **key attributes** critical for understanding cardiovascular health in women, including:  

- **Demographics:** Age, Sex, Income, Country, Continent, Hemisphere  
- **Health Metrics:** Cholesterol, Blood Pressure, Heart Rate, Diabetes, BMI, Triglycerides  
- **Lifestyle Factors:** Smoking, Alcohol Consumption, Exercise Hours Per Week, Diet, Physical Activity, Sedentary Hours, Sleep Hours  
- **Medical History:** Family History of Heart Disease, Previous Heart Problems, Medication Use, Stress Level
  
	• Dataset Name: female_heart_df2
  • Source: Kaggle - [Heart Attack Prediction Dataset!](https://github.com/user-attachments/assets/4eba3d96-36ef-4a7c-8df7-a1ab4b0ab19f)

### **Target and Predictor Variables**  

- **X (Predictors):** Health Metrics, Lifestyle Factors, Medical History  
- **Y (Target):** Female Heart Attack Risk (**1 = At-risk, 0 = No-risk**)  
  - **1,708 no-risk cases**  
  - **944 at-risk cases**  
![image](https://github.com/user-attachments/assets/a536a4c3-cb06-4b77-87d9-2ef562f58a86)

---
## Data Clean Up and Preprocessing

In the data preprocessing phase of this project, several critical steps were undertaken to prepare the dataset for effective analysis and modeling. Initially, the dataset was examined for missing values, which were addressed using appropriate imputation techniques to maintain data integrity. For instance, Numerical Variables: Missing data in numerical variables were imputed using the mean or median values of the respective features. This approach replaces missing entries with the central tendency of the data, preserving the overall distribution and minimizing bias.​ Categorical Variables:  were then transformed into numerical representations through encoding methods, facilitating their use in machine learning algorithms. Numerical features underwent standardization to ensure uniform scaling, thereby enhancing model performance. Additionally, the dataset was scrutinized for duplicate entries, which were removed to prevent data redundancy and potential biases. Feature selection was conducted using correlation analysis to identify and retain the most relevant variables, improving model interpretability and efficiency. Outliers were detected and treated using statistical methods such as the Interquartile Range (IQR) and Z-score techniques, reducing noise and ensuring robust model training. These data cleaning steps ensured a high-quality dataset, essential for developing accurate and reliable predictive models.
![image](https://github.com/user-attachments/assets/0816a471-39f5-429f-8b1d-956723f71b15)

---

## Exploratory Data Analysis (EDA)  

### **Geographic Trends in Smoking**  

- **Highest % of Female Smokers by Continent:**  
  - **Africa:** **68.8%**  
  - **Europe:** **67.7%**  
  - **Asia/South America:** **65.7%**  

- **Top 3 Countries with Most Female Smokers:**  
  - **Italy:** **72.1%**  
  - **South Africa:** **70.9%**  
  - **Vietnam:** **70.3%**  

### **Smoking and Heart Attack Risk**  
Strongest positive correlation: Smoking and Age (0.81) – older women tend to smoke more.

![image](https://github.com/user-attachments/assets/bdcb19f3-82a0-4a67-b3cb-13353dd3b340)

Female smokers over **59 years old** (1,094 individuals) exhibit a high smoking prevalence, while **654 adult females (39-59 years old)** are also affected. Smoking significantly increases heart attack risk.  

- **Survival Rates Without Heart Attack Risk:**  
  - Smokers aged 40-59: **64.83%**  
  - Smokers over 59: **64.81%**  

- **Survival Rates With Heart Attack Risk:**  
  - Smokers aged 40-59: **35.17%**  
  - Smokers over 59: **35.19%**
    
![image](https://github.com/user-attachments/assets/f4b3cdb0-1c03-439b-9668-ca8451df9914)

Additionally, **49.1% of female patients** have a **family history** of heart disease, reinforcing the need for **targeted prevention strategies**.  

### **Other Risk Factors**  

- **Stress levels 5 and 6 are most common**, with stress levels **9 and 10 linked to high smoking rates** (197 and 184 female smokers).  
- **Smoking impacts sleep**, with **69.4% of smokers sleeping only 4 hours** and **68.3% sleeping 10 hours**.  
- **Cholesterol increases with age:**  
  - **19-39 years:** High cholesterol (**456 cases**)  
  - **40-59 years:** High cholesterol (**399 cases**)  
  - **60+ years:** High cholesterol (**643 cases**)  
- **Heart rate categories:**  
  - **Normal:** **1,489 cases**  
  - **Bradycardia (Slow HR):** **786 cases**  
  - **Tachycardia (Fast HR):** **377 cases**  
- **Diabetes is a major risk factor:**  
  - **40-50-year-old diabetics have the highest heart attack risk**  
  - **Non-diabetic seniors (60-85) also face high risk**
    
![image](https://github.com/user-attachments/assets/8d02621c-b7f7-4398-b923-bc5e7396c383)

---

## Machine Learning Models  

Baseline Model

![image](https://github.com/user-attachments/assets/0fdce8aa-75c3-45a6-939c-93b87d89fac5)

Various machine learning models were tested to predict **heart attack risk in women**, with a focus on improving recall for high-risk cases.

The Baseline Logistic Regression model, a widely used statistical model, served as the baseline approach with and without Principal Component Analysis (PCA) to examine the impact of dimensionality reduction. PCA, achieved the highest accuracy (0.6667) but performed poorly in identifying high-risk individuals, with a recall of only 0.01 and an F1-score of 0.02, indicating significant bias toward the majority class.

Tree-based models, including Random Forest (PCA) and Decision Tree, were applied to capture complex feature interactions, showed marginal improvements in recall (0.07 and 0.06, respectively), but their predictive power remained low. 
 
The Random Forest model with SMOTE balancing emerged as the best-performing model, significantly improving recall (0.67) and F1-score (0.66), demonstrating that synthetic oversampling effectively addressed class imbalance. 

The Best GridSearch Random Forest model achieved the highest recall (0.68) but at the cost of lower accuracy (0.6404), making it useful but less optimal than the SMOTE-balanced model.

Other models, such as XGBoost and Tuned Random Forest, performed poorly. XGBoost had the lowest accuracy (0.5932) and recall (0.16), making it the least effective model. Tuned Random Forest achieved similar accuracy to Logistic Regression (0.6629) but had an extremely low recall (0.03), indicating its inability to identify high-risk individuals effectively. Lastly, the Stacking Model achieved the highest accuracy (0.6681) but still had a very low recall (0.03) and F1-score (0.05), making it ineffective for practical use.

These results emphasize the importance of handling class imbalance in heart attack risk prediction. While traditional models like Logistic Regression and Random Forest struggled with low recall, SMOTE balancing significantly enhanced the performance of the Random Forest model, making it the most reliable option for identifying high-risk patients. Future research should explore further optimization techniques, including cost-sensitive learning and hybrid models, to refine prediction performance.


| **Model**                     | **Accuracy** | **Recall (High-Risk)** | **F1-Score (High-Risk)** | **Notes** |
|--------------------------------|-------------|-----------------|------------------|-------------------------|
| **Baseline Logistic Regression (PCA/No PCA)** | **0.6667** | **0.01** | **0.02** | Biased toward Class 0, poor high-risk detection |
| **Random Forest (PCA)** | **0.6478** | **0.07** | -- | Slight recall improvement but still weak |
| **Decision Tree** | **0.6535** | **0.06** | -- | Marginal improvement over Logistic Regression |
| ✅ **Random Forest with SMOTE** | **0.6520** | **0.67** | **0.66** | **Best model; SMOTE balancing significantly improved recall** |
| **Best GridSearch Random Forest** | **0.6404** | **0.68** | -- | Best recall but lower accuracy |
| ❌ **XGBoost** | **0.5932** | **0.16** | -- | **Worst model; lowest accuracy and recall** |
| **Tuned Random Forest** | **0.6629** | **0.03** | -- | Comparable accuracy to Logistic Regression but **very low recall** |
| **Stacking Model** | **0.6681** | **0.03** | **0.05** | **High accuracy but recall is too low to be useful** |

### **Key Findings**  

The **Random Forest with SMOTE model** performed the best, achieving:  
- **Accuracy:** **65.2%**  
- **Precision:** **67%**  
- **Recall:** **66%**  

SMOTE (Synthetic Minority Over-sampling Technique) effectively balanced the dataset, improving recall and ensuring fewer high-risk cases go undetected.  

---

## Metrics

In evaluating the performance of various machine learning models for predicting heart attack risk among female patients, several key metrics were utilized to assess their effectiveness. Accuracy was measured to determine the proportion of correct predictions made by each model. However, given the class imbalance in the dataset, with a higher number of 'no-risk' cases compared to 'at-risk' cases, additional metrics were crucial for a comprehensive evaluation. Recall (Sensitivity) for the 'at-risk' class was calculated to identify the model's ability to correctly detect actual positive cases, reflecting its effectiveness in identifying patients truly at risk. The F1-Score for the 'at-risk' class, which is the harmonic mean of precision and recall, was also computed to provide a balanced measure of the model's performance, especially in scenarios with imbalanced classes. These metrics collectively offered a nuanced understanding of each model's predictive capabilities, guiding the selection of the most appropriate algorithm for accurate heart attack risk prediction in the female cohort.

---
## Feature Importance  

### **Top Contributing Factors to Heart Attack Risk in Women:**  

| **Feature** | **Importance (%)** |
|------------|------------------|
| **Stress Level** | **7.33%** |
| **Sleep Hours Per Day** | **6.18%** |
| **Sedentary Hours Per Day** | **6.17%** |
| **Previous Heart Problems** | **5.95%** |
| **Alcohol Consumption** | **5.91%** |
| **Cholesterol** | **5.88%** |
| **Family History** | **5.87%** |
| **Heart Rate** | **5.66%** |
| **Obesity** | **5.56%** |
| **Diabetes** | **5.51%** |

For this analysis of heart attack risk among female patients, a notable collinearity was observed between the variables smoking and age. This relationship indicates that as age increases, the likelihood of smoking also rises, which can complicate the interpretation of each variable's individual impact on heart attack risk. Despite this collinearity, feature importance assessments from the machine learning models revealed that stress level and previous heart problems were more significant predictors of heart attack risk than smoking and age. This suggests that while smoking and age are related, their combined effect may be less critical in the presence of other dominant risk factors. Addressing collinearity is essential to ensure accurate model interpretations and to identify the most influential predictors for targeted
---

## Conclusion  

This study identifies **key factors contributing to heart attack risk in women**, such as **stress, sleep patterns, sedentary behavior, and previous heart problems**. Addressing these through **lifestyle changes, stress management, and physical activity** can significantly reduce risk. These insights suggest that preventive strategies should focus not only on traditional medical risk factors but also on behavioral and lifestyle interventions.

Future research should:  
- Incorporate **genetic data, diet, and medication use** to refine predictions.  
- **Expand the dataset** for better risk assessment.  
- Further **optimize machine learning models** with cost-sensitive learning techniques.  

This project demonstrates how **machine learning can improve heart attack risk prediction for women**, paving the way for **more personalized, AI-driven healthcare interventions**. 🚀  

---

