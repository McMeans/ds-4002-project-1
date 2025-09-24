# Predicting Product Ratings from Amazon Reviews Using Weighted Keyword Analysis

## Overview
This repository contains a project aimed at predicting Amazon product ratings based on the text of user reviews using a **weighted keyword analysis** approach. The goal is to develop a predictive model with at least 80% accuracy by analyzing the correlation between frequently used words in reviews and their corresponding star ratings. This project can help potential buyers better evaluate products by contextualizing subjective reviews.

Key highlights:  
- **Goal Statement:** Predict product ratings from review text with ≥80% accuracy.  
- **Research Question:** What is the most effective way to assign weightings to words across different product categories to identify biased reviews?  
- **Model Approach:** Linear Support Vector Classifier (SVC) trained on TF-IDF vectorized review text.

---

## Contents of the Repository
The repository includes:  
- **Data preprocessing scripts** – for cleaning and preparing Amazon review data.  
- **Model training and evaluation scripts** – to train a Linear SVC and evaluate predictions.  
- **Exploratory Data Analysis (EDA)** – including plots and statistics of review distributions, keywords, and rating patterns.  
- **Results and Hypothesis Testing** – including predicted vs. actual ratings, mean errors, and t-test analyses.  
- **Documentation** – including data dictionary, references, and instructions to reproduce results.

---

## Software and Platform

**Software Used:**  
- Python 3.10+  
- Jupyter Notebook (optional for exploration)  

**Python Libraries / Packages:**  
- `pandas` – for data manipulation  
- `numpy` – for numerical operations  
- `scikit-learn` – for TF-IDF vectorization and Linear SVC  
- `matplotlib` / `seaborn` – for data visualization  

**Platform:**  
- Windows / Mac / Linux (fully compatible across platforms)  

**Installation:**  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

## Map of Documentation

## Reproducing Results

Follow these steps to reproduce the results of this project:

### Step 1: Obtain Data
- Download the Amazon Fashion review dataset (~2.5 million reviews).  
- Place the dataset in the `data/` folder as `raw_reviews.csv`.  

### Step 2: Preprocess Data
- Run `scripts/preprocess.py` to:
  - Remove incomplete rows.
  - Lowercase text and remove extra spaces.
  - Compute TF-IDF representation of review text.
  - Generate additional features such as review length and emotional word presence.
  - Export cleaned data as `cleaned_reviews.csv`.

### Step 3: Train the Model
- Run `scripts/train_model.py` or `notebooks/model_training.ipynb` to:
  1. Load the cleaned dataset.
  2. Split the dataset into:
     - 70% training
     - 20% validation
     - 10% test
  3. Train a Linear SVC classifier using TF-IDF features.
  4. Apply 5-fold cross-validation for model stability.
  5. Output predicted ratings (`predicted_rating`) for the test set.

### Step 4: Evaluate Model
- Compute evaluation metrics:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix
  - Mean absolute error
- Visualize predictions vs. actual ratings:
  - `results/predicted_vs_actual.png`
- Analyze additional plots (e.g., review length, helpful votes).

### Step 5: Hypothesis Testing
- Separate predictions into:
  - Over-predicted (predicted > actual)
  - Under-predicted (predicted < actual)
  - Correct predictions
- Perform one-sample t-tests on each group to determine if predictive errors differ significantly from zero.
- Conduct a two-sided test on the entire dataset to check for systematic bias.
