# Medical Equipments Cost Prediction

## Overview
This repository contains code and documentation for the Kaggle competition aimed at predicting the cost of medical equipment for different customers using regression models.

## Problem Statement
The goal is to accurately predict the **Cost** of medical equipment given various customer and item features. The evaluation metric is Mean Squared Error (MSE).

## Dataset
- `train.csv`: Training data with features and target cost.
- `test.csv`: Test data with features only.
- `sample_submission.csv`: Format for submission with Customer Id and predicted Cost.

## Project Structure
```
Medical-Equipments-Cost-Prediction/
│
├── linear_regression/
│   ├── linear_regression.ipynb          # Notebook with EDA, preprocessing, and training for Linear Regression
│   ├── linear_regression_submission.csv # Kaggle submission file for this model
│   ├── README.md                        # Notes and observations specific to Linear Regression
│
├── random_forest/
│   ├── random_forest.ipynb              # Notebook for Random Forest model
│   ├── random_forest_submission.csv     # Final predictions CSV for submission
│   ├── README.md                        # Model-specific documentation
│
├── decision_tree/
│   ├── decision_tree.ipynb              # Decision Tree training and evaluation
│   ├── decision_tree_submission.csv     # Submission file for this model
│   ├── README.md
│
├── data/
│   ├── train.csv                       # Training dataset (not usually stored in repo)
│   ├── test.csv                        # Test dataset
│   ├── sample_submission.csv           # Example submission format
│
└── README.md                           # Overall project description and instructions
```
## Approach
- Explored and prepared data using Jupyter notebooks within each model’s folder.
- Implemented a variety of regression models: Linear Regression, Random Forest, Decision Tree, and more, organized separately for clarity.
- Evaluated each model’s performance and submitted predictions to Kaggle to track leaderboard scores.
- Documented findings and parameter tuning in each model’s README file.

## Usage
1. Clone the repo.
2. Download Kaggle dataset files (`train.csv`, `test.csv`) and place them in the `data/` folder.
3. Review notebooks in each model folder to understand training and inference details.
4. Run notebooks to reproduce results.

## Evaluation Metric
- Mean Squared Error (MSE).

## Notes
- Models are limited to those introduced in the initial course part.


