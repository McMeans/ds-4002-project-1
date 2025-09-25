# Predicting Product Ratings from Amazon Reviews Using Weighted Keyword Analysis

## Overview
This repository contains a project aimed at predicting Amazon product ratings based on the text of user reviews using a **weighted keyword analysis** approach. The goal is to develop a predictive model with at least 80% accuracy by analyzing the correlation between frequently used words in reviews and their corresponding star ratings. This project can help potential buyers better evaluate products by contextualizing subjective reviews.

---

## Contents of the Repository
The repository includes:  
- **Software and platform section** – the type(s) of software used for the project, names of any add-on packages that need to be installed with the software, and the platform (e.g., Windows, Mac, or Linux) used
- **A map of our documentation** – an outline or tree illustrating the hierarchy of folders and subfolders contained in your Project Folder and the files stored in each folder or subfolder
- **Instructions for reproducing our results** – explicit step-by-step instructions to reproduce the Results of our study

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
```

## Map of Documentation

In this section, you should provide an outline or tree illustrating the hierarchy of folders and subfolders contained in your Project Folder, and listing the files stored in each folder or subfolder.

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
