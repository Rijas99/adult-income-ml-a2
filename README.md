# Adult Income Classification

This is my Machine Learning Assignment 2 project. I trained five classification models, compared them using different evaluation metrics, and created a small Streamlit app to test the saved models.

## a. Problem Statement

The aim of the project is to predict whether a person's yearly income is `<=50K` or `>50K` using census information such as age, education, occupation, working hours, and marital status. Since there are only two possible output classes, this is a binary classification problem.

## b. Dataset Description

I used the Adult Income dataset, which is also known as the Census Income dataset. The dataset used in this project is stored in `adult.csv`.

- Total records: 48,842
- Input features: 14
- Target column: `income`
- Target classes: `<=50K` and `>50K`
- Data types: numerical and categorical

Some categorical columns contain `?`, which means the value is missing. I replaced these values with proper missing values before preprocessing the data.

The target classes are not balanced. Around 76% of the records belong to the `<=50K` class, so I did not depend only on accuracy while comparing the models.

## Project Workflow

The main steps I followed were:

1. Load and check the dataset.
2. Replace `?` values with missing values.
3. Separate the features and the income target.
4. Split the data into 80% training data and 20% test data using stratified sampling.
5. Fill missing numerical values with the median and scale the numerical columns.
6. Fill missing categorical values with the most frequent value and apply one-hot encoding.
7. Train five classification models.
8. Evaluate each model and save the trained pipelines using Joblib.
9. Build a Streamlit interface for comparing and testing the models.

I used `random_state=42` for the train-test split and the models that support it, so the results can be reproduced.

## c. GitHub Repository Link

https://github.com/Rijas99/adult-income-ml-a2

## d. Models Used and Performance Comparison

I trained and evaluated five classification models:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbours (kNN)
- Gaussian Naive Bayes
- Random Forest

The following results were obtained on the 20% test split:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8524 | 0.9042 | 0.7414 | 0.5885 | 0.6562 | 0.5699 |
| Decision Tree | 0.8141 | 0.7475 | 0.6098 | 0.6198 | 0.6148 | 0.4923 |
| kNN | 0.8337 | 0.8548 | 0.6681 | 0.6061 | 0.6356 | 0.5292 |
| Naive Bayes | 0.6204 | 0.8287 | 0.3794 | 0.9213 | 0.5374 | 0.3866 |
| Random Forest | **0.8602** | 0.9038 | **0.7430** | 0.6356 | **0.6851** | **0.5989** |

## What I Observed

Random Forest gave the best overall performance. It had the highest accuracy, F1 score, and MCC. Logistic Regression was also surprisingly strong and achieved the highest AUC by a very small margin.

Naive Bayes had the highest recall, but its precision and accuracy were low. From this, I understood that it predicted many people as earning above 50K, including many incorrect cases. The single Decision Tree performed reasonably well but was weaker than Random Forest, most likely because one unrestricted tree can overfit the training data.

kNN gave average results overall, with an accuracy of 0.8337, AUC of 0.8548, F1 score of 0.6356, and MCC of 0.5292. It performed slightly better than Logistic Regression in recall, but Random Forest still performed better in all the main metrics. One possible reason is the large number of features created after one-hot encoding. For example, the `native-country` column alone has more than 40 categories. Since kNN compares samples based on distance, having too many dimensions can make it harder to identify which samples are actually close to each other.

For this dataset, I consider Random Forest the best model because it gave the most balanced result. MCC was useful here because the income classes are imbalanced and accuracy alone can be misleading.

## Streamlit App

The Streamlit app has two sections:

- A comparison page showing the saved metric table and a bar chart for all five models.
- A model explorer where `test_data.csv` can be uploaded, a model can be selected, and its metrics, confusion matrix, and classification report can be viewed.

## How to Run the Project

### 1. Open the project folder

```bash
cd rijas-ml-a2-income
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python train_models.py
```

This command creates the trained model files inside the `model` folder, saves the comparison metrics, and generates `test_data.csv`.

### 5. Start the Streamlit app

```bash
streamlit run app.py
```

After the app opens, `test_data.csv` can be uploaded in the model explorer section.

## Project Structure

```text
rijas-ml-a2-income/
|-- adult.csv
|-- app.py
|-- check_data.py
|-- requirements.txt
|-- train_models.py
|-- test_data.csv
|-- README.md
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- random_forest.joblib
    |-- label_encoder.joblib
    `-- metrics_comparison.csv
```

## Files in This Project

- `check_data.py` performs a quick check of the dataset shape, columns, target distribution, and sample rows.
- `train_models.py` handles preprocessing, model training, evaluation, and saving the results.
- `app.py` contains the Streamlit user interface.
- `test_data.csv` is the held-out test set generated during training.
- `model/metrics_comparison.csv` contains the final metric values used by the app.

## Tools and Libraries

Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn, and Joblib were used in this project.

## Conclusion

This project helped me understand why multiple evaluation metrics are important for an imbalanced classification problem. It also gave me practice in building a full ML workflow where preprocessing and the trained classifier are saved together as one pipeline. Among the five models, Random Forest gave the best balanced performance for predicting adult income.
