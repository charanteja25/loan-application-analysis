# 📊 Loan Default Risk Prediction

## 📝 Overview
This project aims to build machine learning models to predict the risk of loan defaults using a dataset containing various customer attributes. The analysis involves extensive data preprocessing, exploratory data analysis, feature selection, and model building to identify customers at high risk of defaulting on their loans.

## 🛠️ Tools and Steps
**Tools:** 🐍 Python, 📊 Pandas, 📈 NumPy, 📊 Matplotlib, 📊 Seaborn, 📊 scikit-learn, 📊 imbalanced-learn, 📊 XGBoost, 📊 SciPy.

**Steps:**
- ### 📈 Data Preprocessing
   - Handling missing values (if any)
   - Feature engineering (e.g., encoding categorical variables)
   - Scaling numerical features
- ### 📊 Imbalanced Data Treatment
   - Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance
- ### 📊 Data Splitting into Train and Test Sets
   - Stratified sampling to maintain class balance
- ### 📊 Feature Selection
   - Recursive Feature Elimination (RFE) to select the most relevant features
- ### 📝 Model Building and Evaluation
   - Logistic Regression
   - Random Forest
   - Decision Tree
   - Gradient Boosting
   - XGBoost
   - AdaBoost
   - Stacking Classifier
   - Evaluation metrics: F1 score, classification report
- ### 📝 Model Selection and Saving
   - Selecting the best-performing model based on performance metrics and overfitting considerations
   - Saving the trained models for future use

## 📈 Data Preprocessing
The dataset was loaded, and any necessary data preprocessing steps were performed, such as handling missing values (if any), feature engineering (e.g., encoding categorical variables), and scaling numerical features using StandardScaler.

## 📊 Imbalanced Data Treatment
The Synthetic Minority Over-sampling Technique (SMOTE) was applied to address the class imbalance in the target variable (loan defaulter), improving the models' ability to predict the minority class.

## 📊 Data Splitting into Train and Test Sets
The dataset was split into training and test sets using stratified sampling to maintain the class balance of the target variable in both sets.

## 📊 Feature Selection
Recursive Feature Elimination (RFE) was used to select the most relevant features for model building, improving model performance and reducing overfitting. The selected features were:

- `REGION_POPULATION_RELATIVE`
- `DAYS_BIRTH`
- `DAYS_EMPLOYED`
- `DAYS_REGISTRATION`
- `DAYS_ID_PUBLISH`
- `OWN_CAR_AGE`
- `AMT_REQ_CREDIT_BUREAU_WEEK`
- `AMT_REQ_CREDIT_BUREAU_MON`
- `AMT_REQ_CREDIT_BUREAU_QRT`
- `AMT_REQ_CREDIT_BUREAU_YEAR`
- `LOG_AMT_INCOME_TOTAL`
- `FLAG_DOC_SCORE`
- `LOG_AMT_CREDIT`
- `LOG_AMT_ANNUITY`
- `EXT_SOURCE_AVG`
- `REGION_RATING_CLIENT_3`
- `DEF_60_CNT_SOCIAL_CIRCLE_1.0`

## 📝 Model Building and Evaluation
Various machine learning algorithms were employed, including Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, XGBoost, AdaBoost, and Stacking Classifier. The models were trained on the training set and evaluated on the test set using appropriate metrics, such as F1 score and classification report.

Here are the F1 scores for each model on the test set:

- Logistic Regression: 0.446
- Random Forest: 0.884
- Decision Tree: 0.906
- Gradient Boosting: 0.912
- XGBoost: 0.949
- AdaBoost: 0.860
- Stacking Classifier: 0.952

The XGBoost and Stacking Classifier models performed the best, with F1 scores of 0.949 and 0.952, respectively.

## 📝 Model Selection and Saving
The Stacking Classifier model was selected as the final model due to its high F1 score on the test set (0.952) and minimal overfitting compared to other models. However, if the client prefers the highest-scoring model, the XGBoost model (test F1 score: 0.949) can be considered, although it may have lower interpretability.

The trained models were saved for future use and deployment.

## 📝 Conclusion
This project demonstrates the process of building and evaluating machine learning models to predict loan default risk using a customer dataset. The README file provides an overview of the project, the tools and steps involved, data preprocessing, imbalanced data treatment, data splitting, feature selection, model building and evaluation, model selection, and saving the trained models.
