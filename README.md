# Predicting Scientific Paper Citations with Machine Learning

This repository contains a machine learning solution for predicting the number of citations received by scientific papers based on their metadata. The project was part of a challenge to build a robust regression model, leveraging text-based and numerical features to generate accurate predictions.

# Project Overview
The goal of this challenge was to develop a model that predicts the number of citations for scientific papers using metadata such as title, abstract, authors, venue, and other related attributes. The provided dataset included training data with citation counts and test data without citation counts, which required predictions.

# Solution Approach
## 1. Data Preprocessing and Feature Engineering
  * Extracted numerical features such as:
    - year: Publication year
    - num_authors: Number of authors
    - num_references: Number of references cited
    - paper_age: Age of the paper based on the year
    - title_word_count: Word count of the title
  * Encoded categorical features like venue using LabelEncoder.
  * Transformed text-based features (title, abstract, authors, venue) using TF-IDF to generate meaningful vectors for model training.
  
## 2. Model Development
  * Explored multiple regression algorithms:
    - Ridge Regression
    - Gradient Boosting
    - LightGBM
    - XGBoost
  * Performed hyperparameter tuning using:
    - GridSearchCV for Gradient Boosting
    - RandomizedSearchCV for LightGBM and XGBoost
  * Selected LightGBM as the final model due to its superior performance and efficiency.


## 3. Model Pipeline
  * Preprocessing: Applied TF-IDF transformations for text features and scaled numerical features directly.
  * LightGBM Hyperparameters:
    - num_leaves=31
    - n_estimators=200
    - max_depth=5
    - learning_rate=0.1
   
  
## 4. Evaluation
  * Used cross-validation to ensure model robustness and reduce overfitting.
  * Achieved the following performance:
    - Training Mean Absolute Error (MAE): 29.81
    - Validation Mean Absolute Error (MAE): 30.20
   
    
# Key Results
  * LightGBM was the best-performing model, outperforming other approaches in terms of accuracy and efficiency.
  * Feature engineering and the inclusion of text-based transformations significantly enhanced the predictive power of the model.

# Conclusion
This project demonstrates the effective use of feature engineering and hyperparameter optimization in building a machine learning model for citation prediction. The LightGBM model, combined with TF-IDF and numerical feature extraction, provided robust predictions, making it a strong solution to the challenge.
