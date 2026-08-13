# check_data.py
# Purpose: confirm the dataset loads correctly before we build models.

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "adult.csv")

# load the dataset
df = pd.read_csv(DATA_FILE)

# 1. how big is it?
print("Rows and columns:", df.shape)

# 2. what are the column names?
print("\nColumns:")
print(list(df.columns))

# 3. what does the target look like?
print("\nTarget 'income' values:")
print(df["income"].value_counts())

# 4. peek at the first few rows
print("\nFirst 3 rows:")
print(df.head(3))
