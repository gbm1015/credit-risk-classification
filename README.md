# Credit Risk Classification Challenge 

## Background

Using a dataset of historical lending activity, from a peer-to-peer lending services company, and leveraging machine learning techniques to train and evaluate a model that can predict loan risk.  The machine learning classification model(s) is developed to identify the creditworthiness of the borrowers.

### Before opening the starterCode folder

1. I created a new repository in GitHub for this project called `credit-risk-classification`. 
2. Inside the new repository I cloned the new repository to my computer.
3. Inside my local Git repository, I created a folder titled "Credit_Risk" and added the starter Jupitor Notebook code credit_risk_classification.ipynb and the folder "Resources" that contains the dataset lending_data.csv.

## Overview of the Analysis

1. The dataset lending_data.csv contains loan related features (variables) about 77,536 borrowers.  75,036 of the loans are identified as low-risk loans (loan_status=0) and the remaining 2,500 loans are identified as high-risk loans (loan_status=1).  The purpose of the analysis is to build a machine learning model that would predict the classification of loans into "low-risk" vs. "high-risk" loans; that is, to predict the loan_status.
2. The dataset also contained the following loan and borrowers' related features (variables) to be included as potential predictors in the logistic regression model. Those variables included:
   - The size of the loan.
   - The interest rate for the loan.
   - The borrower's income.
   - The borrower's total debt to income ratio.
   - The number of accounts the borrower has.
   - Presence of any derogatory marks against the borrower (0/1 flag).
   - The borrower's total debt amount.
3. Two different models were developed:
   (i) Logistic regression (#1) with the original dataset lending_data.csv, that included 75,036 low-risk loans and 2,500 high-risk loans.
   (ii) Logistics regression (#2) with oversampling the high-risk loans using RandomOverSampler module from imbalanced-learn, which generated a new dataset of 56,277 low-           risk loans (undersamdpling from the low risk loans) and 56,277 high-risk loans (oversampling from the high-risk loans.
