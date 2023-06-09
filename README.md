# Credit Risk Assesment Using Machine Learning Model

# Introduction

The financial industry heavily relies on assessing credit risk to determine the likelihood of a borrower defaulting on their loan. Traditional methods have limitations in terms of accuracy and efficiency. With the advent of machine learning algorithms, there is an opportunity to improve the accuracy of credit risk assessment models. The goal of this project is to develop a machine learning model that can accurately predict the credit risk of applicants based on their attributes.

This project will involve exploring and analyzing a dataset (“credit_risk_customers” taken from kaggle) of a customer credit information, preparing the data for ML models, and then developing and evaluating various models such as logistic regression, decision trees, random forests, and support vector machines.

By accurately predicting credit risk, financial institutions can make more informed decisions on lending and minimize the risk of default.

# Dataset Description:

The “Credit Risk Customer” dataset contains 1,000 observations of credit applications with 21 attributes used to determine whether the applications were approved or denied. The attributes include demographic information such as age, gender, and incomes, and credit-related information such as credit history, savings status, loan amount. Each row in the dataset represents an individual credit application, and the “class” column indicates whether the application was classified as “good” or “bad” credit risk.

Checking_status: status of existing checking account

Duration: duration in months

Credit_history: credits taken, paid back duly, delays, critical accounts

Purpose: purpose of the credit

Credit_amount: amount of credit

Savings_status: status of savings account/bond

Employment: present employment, in number of years

Installment_commitment: installment rate in percentage of disposable income

Personal_status: sex and marital data

Other_parties: other debtors/guarantors

Residence_since: in years

Property_magnitude: car, real estate, insurance, etc

Age: in years

Other_payment_plans: other payment methods

Housing: rent, own, for free, etc

Existing_credits: number of existing credits

Job: job classifications: skilled, unskilled, etc

Num_dependents: number of dependents

Own_telephone: yes, or none

Foreign_worker: yes or no

Class: good or bad

![image](https://github.com/shailw/credit_risk/assets/96182727/cf05304f-91af-4e59-9aa4-6896cb4053ad)
<img width="779" alt="Screenshot 2023-06-08 at 11 44 53 PM" src="https://github.com/shailw/credit_risk/assets/96182727/598a6b99-9b30-429a-b610-ec663fd806ef">


# Project Scope
The scope of this project includes the following:

Exploratory Data Analysis (EDA): Perform an analysis of the "Credit Risk Customer" dataset to understand its structure, identify missing values, and explore the relationship between variables. Visualize the data using graphs and statistical measures.

Data Preprocessing: Preprocess the dataset by handling missing values and duplicates. Encode categorical columns using one-hot encoding. Scale the data by normalizing the values. Apply oversampling techniques to balance the dataset.

Feature Modeling: Build several machine learning models, including logistic regression, decision trees, random forests, and support vector machines, to predict the credit risk of applicants based on their attributes.

Model Evaluation: Evaluate the performance of each model using various metrics such as accuracy, precision, recall, F1-score, and AUC-ROC. Compare the performance of different models to identify the best-performing model.

Model Selection: Select the best-performing model based on the evaluation metrics. Fine-tune the selected model's hyperparameters to improve its performance.

# Exploratory Data Analysis
Perform exploratory data analysis on ‘credit risk customer’ dataset to understand the dataset’s structure, identify missing values, and explore the relationship between variables.

Data Preprocessing: preprocessing the dataset by clearing the data, transforming, and normalizing the data, and handling missing values, as necessary.

Feature Modeling: to build several machine learning models to predict the credit risk of applicants based on their attributes.

Model Evaluation: evaluate the performance of each model using various metrics such as accuracy, precision, recall, and F1 score.

Model Selection: select the best performance model and fine-tune its hyperparameters to improve its performance.

# EDA
For the EDA we started by loading the dataset into the Jupyter notebook and then we got know the number of columns and their types such as numerical and categorial. We also identified the data types of each columns and processed the data type so that it can be used in the Machine learning model.

We identified the number of missing values in each column to get to understand the data set more accurately and to decide how to proceed further with the null values.

We also plotted graphs between the various categories based on varied factors, plotted correlation matrix and Boxplots. we also dervied Target variable Distribution between good and bad score, Metric variable Distribution like duration distribution,installment commitment distribution, credit amount distribution and categorical variable distribution like checking status distribution, saving status distribution and personal status distribution using Bar Graph.

![image](https://github.com/shailw/credit_risk/assets/96182727/2f2146d9-89fd-4cc1-bf21-6ae795459198)
<img width="700" alt="Screenshot 2023-06-08 at 11 45 13 PM" src="https://github.com/shailw/credit_risk/assets/96182727/d7891746-6b56-4e8a-b2e7-89d3663b1c98">
<img width="1119" alt="Screenshot 2023-06-08 at 11 45 55 PM" src="https://github.com/shailw/credit_risk/assets/96182727/34a3c15d-9f53-462e-b4f9-9e19f08b5ffb">
<img width="1091" alt="Screenshot 2023-06-08 at 11 46 51 PM" src="https://github.com/shailw/credit_risk/assets/96182727/39282621-9a69-491e-a216-c954e91da2f5">
![image](https://github.com/shailw/credit_risk/assets/96182727/959157ff-b98b-4856-93ba-968116efa0fd)

# Data Preprocessing
In Data Preprocessing we used OneHotEncoder for categorical columns like Checking_status, purpose, saving_status to replace it so that can be further used for machine learning algorithm and create a updated dataset.

Checked for missing values and duplicates in the complete dataset and removed them accordingly.

We scaled the data by normalizing the values and oversampling it for the selected columns as the normalized data gives better results.

<img width="1122" alt="Screenshot 2023-06-09 at 12 03 18 AM" src="https://github.com/shailw/credit_risk/assets/96182727/eec4ec63-7e28-46e6-b326-27256592230d">


# Feature and Evalute Modeling
For modeling we split the new dataset for testing and training.

We implemented Logical Regression, Random Forest and Decision Tree model to find various metrics like Accuracy, Precision, Recall, F1-Score, AUC-ROC for each model to find out which model suits the best for the dataset and provides better output.

We also used the Model Ranking to predict the best of all the implemented model for each metrics.

![image](https://github.com/shailw/credit_risk/assets/96182727/6ec2cef2-c38b-4851-9be9-68d23d0fefad)

# Model Selection:
Based on the results of the model rankings of the different metrics we got to know that Logistic Regression suits better for this data set and gives better accuracy when compared with other models.

We tuned the data with hyperparameters using GridSearchCV. With that tuned dataset obtained from performing grid search cross validation we designed the final model.

We evaluated the Final model performance with different metrics like accuracy, precision, recall, F1-Score, AUC-ROC for all the three models.

Even after tuning the data we found out that the Logical Regression and Random Forest Classifier models provided similar level of accuracy and Decision Tree Classifier produced little less when compared other two models.

![image](https://github.com/shailw/credit_risk/assets/96182727/790062c7-a3be-40e4-8fe1-5504668471cb)
![image](https://github.com/shailw/credit_risk/assets/96182727/08e07d9e-b01e-482e-9181-5e8b856370d8)
![image](https://github.com/shailw/credit_risk/assets/96182727/2e9ff3c4-b66c-4d9b-91ee-66e262237806)

# Project Out of Scope
The following aspects are out of scope for this project:

Data Collection: The project assumes that the "Credit Risk Customer" dataset is already available and does not involve the collection of new data.

Feature Engineering: The project does not include extensive feature engineering techniques such as creating new features or transforming existing features based on domain knowledge.

Model Interpretability: The project does not focus on interpreting the models' internal workings and understanding the specific factors driving their predictions. The emphasis is on model performance and accuracy.

Deployment: Deploying the selected model into a production environment or creating a user interface for interaction with the model is beyond the scope of this project.

Exploring Additional Model Algorithms: While logistic regression, decision trees, random forests, and support vector machines are included in the scope, exploring other model algorithms or more advanced techniques like ensemble methods or deep learning architectures is not covered.

The project's primary goal is to develop a credit risk assessment model using machine learning techniques and evaluate its performance. It provides a foundation for further development and customization based on specific requirements and domain expertise.

# Conclusion
This project demonstrates the development of a machine learning model for credit risk assessment. By accurately predicting credit risk, financial institutions can make more informed lending decisions and minimize the risk of default. The project includes exploratory data analysis, data preprocessing, feature modeling, model evaluation, and model selection. The logistic regression model was selected as the best-performing model and was fine-tuned using hyperparameters. The model's performance was evaluated using various metrics. The project scope does not cover data collection, feature engineering, deployment, exploring additional model algorithms, or interpretability of the models. The project provides a foundation for building an effective credit risk assessment model and can be further extended and enhanced based on specific requirements and domain expertise.
