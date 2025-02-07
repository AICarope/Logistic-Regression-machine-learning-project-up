Update me again
# Capstone Project: Heart Attack Risk Prediction in Female Populations
##🚧 In Progress

Developing a **Logistic Regression machine learning model to predict heart attack risk in female populations**. The focus is on identifying key predictive factors such as age, cholesterol, blood pressure, smoking, obesity, and stress and understanding their interrelationships.
Using a dataset from Kaggle, containing 8,763 patient entries, with 2,652 female patients, the goal is to accurately predict heart attack risk based on health metrics, lifestyle factors, and medical history.
The model is evaluated using Python, applying metrics such as accuracy, precision, recall, and F1-score, with cross-validation and hyperparameter tuning to optimize performance. By tailoring this model to women’s unique cardiovascular risk factors, this project seeks to improve early detection, personalized treatment, and clinical practices related to heart disease in women.

## Materials
You can access the materials by clicking

-[Python Notebook: Exploratory Data Analysis(EDA)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/2_Women_EDA.ipynb)

-[Python Notebook: Machine Learning (ML)](https://github.com/AICarope/Logistic-Regression-machine-learning-project-up/blob/main/3_Women_ML.ipynb)


## 📊 Project Visualization
<img src="https://github.com/user-attachments/assets/43b4979f-ea8a-435c-b2c2-84231939164c" alt="Graph" width="500"/>


## 📂 Dataset Information

- **Dataset Name:** `female_heart_df2`
- **Source:** [Kaggle - Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)
- **Target Variable:** `Heart Attack Risk` (1 = Yes, 0 = No)
- **Predictor Variables:**
  -  **Health Metrics:** Cholesterol, Blood Pressure, Heart Rate, BMI, Triglycerides
  -  **Lifestyle Factors:** Smoking, Alcohol Consumption, Exercise Hours, Diet, Sleep Hours, Sedentary Hours
  -  **Medical History:** Diabetes, Previous Heart Problems, Medication Use, Family History
  -  **Demographics:** Age, Income, Country


## 📈 Key Insights from Exploratory Data Analysis (EDA)

### **Demographics & Risk Distribution**
-  **Seniors (59+)** make up **41.25%** of the dataset.
-  **Smoking & Age** show a **strong correlation (0.81)**.
-  **Cholesterol & Heart Attack Risk** correlation is **weak (0.04)**.
-  **Diabetes is present in 64.9%** of the dataset but has a **weak correlation (0.03)**.
-  **Obesity:** **49.9%** of females in the dataset are classified as **obese**.

### **Lifestyle & Health Patterns**
-  **Physical Activity:** Most engage in **1-3 days/week** of exercise.
-  **Sedentary Behavior:** **25% of female smokers** sit for **9+ hours/day**.
-  **Sleep Patterns:** The majority sleep between **4-10 hours/day**, with **8 hours** being most common.

### **Geographic Trends**
- **Highest % of Female Smokers by Continent:**
  -  **Africa:** **68.8%**
  -  **Europe:** **67.7%**
  -  **Asia/South America:** **65.7%**

- **Top 3 Countries with Most Female Smokers:**
  - 🇮🇹 **Italy:** **72.1%**
  - 🇿🇦 **South Africa:** **70.9%**
  - 🇻🇳 **Vietnam:** **70.3%**


##🤖 Machine Learning Model Performance

Class 1 = Heart Attack Risk
Class 0 = No Heart Attack Risk

✅ Best Model: Random Forest with SMOTE


## 🤖 Machine Learning Model Performance


| 🏷️ Model                                     |  Accuracy |  Recall (Class 1) |  F1-Score (Class 1) |  Interpretation |
|----------------------------------------------|------------|---------------------|----------------------|----------------------------------------------------------------|
| **Baseline Logistic Regression (With/Without PCA)** | 0.6667 | 0.01 | 0.02 | Highly biased towards Class 0; fails to detect high-risk individuals. |
| **Random Forest (PCA)**                     | 0.6478 | 0.07 | -- | Slight recall improvement but still too low to be useful. |
| **Decision Tree**                            | 0.6535 | 0.06 | -- | Slight recall improvement but not significantly better than Logistic Regression. |
| **Random Forest with SMOTE** ✅              | **0.6520** | **0.67** | **0.66** | Best model; SMOTE balancing improved recall significantly. |
| **Best GridSearch Random Forest**           | 0.6404 | 0.68 | -- | Best recall but lower accuracy; useful but less optimal. |
| **XGBoost**                                  | 0.5932 | 0.16 | -- | Worst model; low accuracy and recall. |
| **Tuned Random Forest**                     | 0.6629 | 0.03 | -- | Comparable accuracy to Logistic Regression but very low recall. |
| **Stacking Model**                           | **0.6681** | 0.03 | 0.05 | Higher accuracy but recall is too low to be useful. |


⚠️ Worst Models:

Logistic Regression → Fails to detect heart attack risk (Recall = 0.01).

XGBoost → Lowest accuracy (0.5932) and weak recall (0.16).

🔬 Alternative Choice:

Best GridSearch Random Forest if recall (0.68) is prioritized, but has slightly lower accuracy.

**Features Used in the Best Model (Random Forest with SMOTE)**

## 🔍 Feature Importance (Top 10 Features) 

|  Original Feature          |  Importance (%) |
|------------------------------|------------------|
| **Stress Level**             | 7.33%            |
| **Sleep Hours Per Day**      | 6.18%            |
| **Sedentary Hours Per Day**  | 6.17%            |
| **Previous Heart Problems**  | 5.95%            |
| **Alcohol Consumption**      | 5.91%            |
| **Cholesterol**              | 5.88%            |
| **Family History**           | 5.87%            |
| **Heart Rate**               | 5.66%            |
| **Obesity**                  | 5.56%            |
| **Diabetes**                 | 5.51%            |

## 🔍 Key Findings on Feature Importance

-  **Top Predictive Features in the Dataset**:
  - **Stress Level** (**7.33%**) is the most important predictor.
  - **Sleep Hours Per Day** (**6.18%**) and **Sedentary Hours Per Day** (**6.17%**) follow closely.
  
-  **Smoking is Not a Key Predictor**:
  - Smoking was **not selected** as a top-ranked feature by the model.
  - The lowest-ranked features, **Obesity (5.56%)** and **Diabetes (5.51%)**, still had **higher importance scores** than Smoking.
  - While **Smoking correlates with Age**, it was **not a direct driver** of heart attack risk.
  - **Other health factors (e.g., Stress, Cholesterol, Sleep, Sedentary Behavior)** contributed more predictive value.

-  **Why Smoking Was Not a Key Feature**:
  - The dataset was transformed using **PCA**, which combined **Smoking** with other features into **principal components**.
  - **SHAP Correlation** measures **linear relationships**, while **Random Forest & tree-based models** capture **complex, nonlinear interactions**.

---

## 📜 **Conclusions & Recommendations**  

 **Key Risk Factors**:  
**Stress Level, Sleep Hours Per Day, Sedentary Hours Per Day, Previous Heart Problems.**  

 **Lifestyle Modifications**:  
Increasing **physical activity**, reducing **sedentary hours**, and **managing stress** can help lower heart attack risk.  

 **Further Research Needed**:  
Investigate the impact of **genetic predisposition, diet, and medication use** as additional risk factors.  

 **Machine Learning Recommendations**:  
**Random Forest with SMOTE** is the best-performing model.  
 **Further hyperparameter tuning** could improve performance and refine predictions.  
