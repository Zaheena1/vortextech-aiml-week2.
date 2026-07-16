# VortexTech AI/ML Internship — Week 2: Build a Classification Model

## What this is

A binary classification model predicting Titanic passenger survival (survived / 
did not survive), built as the Week 2 task for the VortexTech AI/ML Internship Track.

The notebook covers:
- Loading and reusing the cleaned Titanic dataset from Week 1
- Selecting a binary target (`survived`) and feature columns (class, sex, age, 
  family size, fare, port of embarkation)
- Converting categorical features (`sex`, `embarked`) into numeric form using 
  `pd.get_dummies()`
- Splitting the data into training (80%) and testing (20%) sets
- Training a Logistic Regression model with scikit-learn
- Evaluating performance with accuracy, precision, recall, and F1-score
- Training a Decision Tree as a comparison model
- A written interpretation of the results and ideas for improvement

## Dataset

`titanic.csv` — the same cleaned Titanic dataset used in Week 1, loaded from the 
public [seaborn-data repository](https://github.com/mwaskom/seaborn-data).

## Results

| Model | Accuracy | F1-score |
|---|---|---|
| Logistic Regression | ~0.80 | ~0.75 |
| Decision Tree | ~0.75 | ~0.69 |

Logistic Regression outperformed the Decision Tree on this dataset. See the 
notebook's final markdown cell for full interpretation.

## Files

- `Week2_Classification_Model.ipynb` — the main notebook
- `README.md` — this file

## How to run it

**Option 1 — Google Colab (recommended):**
Open `Week2_Classification_Model.ipynb` in Colab and run all cells top to bottom.

**Option 2 — Locally:**
```bash
pip install pandas scikit-learn
jupyter notebook Week2_Classification_Model.ipynb
```
Then run all cells top to bottom.
