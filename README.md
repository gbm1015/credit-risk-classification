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
   - Logistic Regression Model (#1) with the original dataset lending_data.csv, that included 75,036 low-risk loans and 2,500 high-risk loans.  The dataset was split into 2      groups (75% to 25% split); the training dataset for building the model with 58,152 borrowers that included 56,277 low-risk loans and 1,875 high-risk loans, and the          test dataset with 19,384 borrowers that included 18,759 low-risk loans and 625 high-risk loans.
   - Logistics Regression Model (#2) with oversampling the high-risk loans in the training dataset using RandomOverSampler module from imbalanced-learn, which generated a        new dataset of 56,277 low-risk loans (same as what was included in the Logistic Regression Model #1 training dataset) and 56,277 high-risk loans (oversampling the # of      high-risk loans from the original 1,875 that was included in the Logistic Regrsssion #1 training dataset).

4. The following steps were implemented for building both Logistics Regression models:
   - Fit a logistic regression model by using the training dataset (x_train and y_train).
   - Save the predictions for the testing data labels by using the testing feature data (x_test) and the fitted model.
   - Evaluate the model's performance by generating a confusion matrix and printing the classification report.

5. Logistic Regression Model (#1) Performance Results:
   - Precision in predicting low-risk loans = 100%.  Precision in predicting high-risk loans = 87%.
   - Accuracy = 94.4%
   - Recall in predicting low-risk loans = 100%.  Recall in predicting high-risk loans = 89%.
  
   Logistic Regression Model (#2) Performance Results:
   - Precision in predicting low-risk loans = 100%.  Precision in predicting high-risk loans = 87%.
   - Accuracy = 99.6%
   - Recall in predicting low-risk loans = 100%.  Recall in predicting high-risk loans = 100%.
     
## Overview of the Prediction Analysis

The Logistic Regression Model (#1) predicts a healthy (low-risk) loan with 100% precision, while it predicts a high-risk loan with a lower precision at 87%. In general,  that logistic regression model is good at predicting whether a loan may default (not a healthy loan, or is a high risk loan) because of its high balanced accuracy at 94.4% and somewhat high f-1 and recall scores. If the bank is still getting a high precision and recall on the test dataset (even if they are lower scores than for the training dataset), it is a good indication about how well the model is likely to perform in real life.  Consequently, the accuracy of the logistic regression model seems to be good enough to start exploring this algorithm in a bank setting for assessing the creditworthiness of borrowers; however, it may be prudent for the bank to start running a pilot with new data to assess the model's reliability on data the model has not "seen" yet.   

The resampled Logistic Regression Model (#2), using the RandomOverSampler module, predicts a healthy/low-risk loan with 100% precision, while it predicts a high-risk loan with a lower precision at 87%; both precision percentages were the same as in the original logistic regression model. However, the balanced accuracy for the resampled logistic regression model is 99.6%, in comparison to 94.4% for the original logistic regression model.  Similarly, the f-1 and recall scores were higher in the resampled logistic regression model.

Therefore, If the goal of the model is to determine the likelihood of high-risk loans, neither models result in above 90% precision score. However, the Logistic Regression Model (#2) results in fewer false predictions for the testing data and would be the beter model to use based on its high accuracy and recall scores.
