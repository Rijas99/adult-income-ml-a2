# Adult Income Classification — ML Assignment 2

M.Tech AIML, BITS Pilani WILP — Machine Learning, Assignment 2

## a. Problem Statement

Predict whether a person's annual income is above or below $50K based on
census attributes such as age, education, occupation, and hours worked per
week. This is a binary classification problem.

## b. Dataset Description

- Source: Adult Income dataset (Kaggle / UCI Census Income)
- File used: `adult.csv`
- Instances: 48,842
- Features: 14 (age, workclass, fnlwgt, education, educational-num,
  marital-status, occupation, relationship, race, gender, capital-gain,
  capital-loss, hours-per-week, native-country)
- Target: `income` — `<=50K` or `>50K`
- Class balance: about 76% `<=50K` vs 24% `>50K`

[Write 2-3 sentences here in your own words about what stands out in the
data — e.g. the class imbalance, the `?` missing values in workclass /
occupation / native-country, or which features you'd expect to matter most.]

## c. GitHub Repository Link

https://github.com/Rijas99/adult-income-ml-a2

## d. Models Used — Comparison Table

Computed on a 20% held-out test split (`test_data.csv`), from
`model/metrics_comparison.csv`:

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8524 | 0.9042 | 0.7414 | 0.5885 | 0.6562 | 0.5699 |
| Decision Tree | 0.8141 | 0.7475 | 0.6098 | 0.6198 | 0.6148 | 0.4923 |
| kNN | 0.8337 | 0.8548 | 0.6681 | 0.6061 | 0.6356 | 0.5292 |
| Naive Bayes | 0.6204 | 0.8287 | 0.3794 | 0.9213 | 0.5374 | 0.3866 |
| Random Forest (Ensemble) | 0.8602 | 0.9038 | 0.7430 | 0.6356 | 0.6851 | 0.5989 |

## Observations

| ML Model | Observation about performance |
|---|---|
| Logistic Regression | [your words] |
| Decision Tree | [your words] |
| kNN | [your words] |
| Naive Bayes | [your words] |
| Random Forest | [your words] |
| **Overall winner** | [your pick + why, referencing AUC/MCC since the classes are imbalanced] |
