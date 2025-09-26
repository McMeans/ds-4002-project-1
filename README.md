# Predicting Product Ratings from Amazon Reviews Using Weighted Keyword Analysis

A project for UVA's DS 4002 course aimed at predicting Amazon product ratings based on user review text using weighted keyword analysis.

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

```
DS-4002-PROJECT-1/
│
├─ data/
│   ├─ amazon_fashion.csv
│   └─ Amazon_Fashion.json
│   └─ README.md
│
├─ output/
│   └─ 1_distribution.png
│   └─ 2_percentages.png
│   └─ 3_pairwise.png
│   └─ final_data.csv
│
├─ scripts/
│   ├─ analysis.ipynb
│   └─ cleaning.ipynb
│   └─ hypothesis_testing.ipynb
│
├─ .gitattributes
├─ .gitignore
├─ LICENSE
└─ README.md
```

## Reproducing Results

Follow these steps to reproduce the results of this project:

# Steps to Reproduce Results

## Prerequisites
- Python 3.x with Jupyter Notebook
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
- Access to the Amazon Fashion review dataset

## Step 1: Data Acquisition and Setup
1. **Clone the ds-4002-project-1 repo locally** – may need to use GitHub LFS due to the large size of the data files:
```
git lfs install
```
   - `Amazon_Fashion.json` - the raw file downloaded
   - `amazon_fashion.csv` - the cleaned dataset with relevant columns used for analysis

1. **Verify project structure** matches the documentation tree:

```
DS-4002-PROJECT-1/
│
├─ data/
│   ├─ amazon_fashion.csv
│   └─ Amazon_Fashion.json
│   └─ README.md
│
├─ output/
│   └─ final_data.csv
│
├─ scripts/
│   ├─ analysis.ipynb
│   └─ cleaning.ipynb
│   └─ hypothesis_testing.ipynb
│
├─ .gitattributes
├─ .gitignore
├─ LICENSE
└─ README.md
```

## Step 2: Exploratory Data Analysis
1. **Open `scripts/analysis.ipynb`** in Jupyter Notebook

2. **Load the cleaned dataset** from `data/amazon_fashion.csv`

3. **Conduct exploratory analysis:**
   - Distribution of star ratings
   - Review length analysis
   - Text feature analysis
   - Correlation between variables

## Step 3: Model Training and Prediction
1. **Continue in `scripts/analysis.ipynb`** or create separate modeling section:

2. **Prepare data for modeling:**
   - Split dataset into train (70%), validation (20%), and test (10%) sets
   - Use TF-IDF vectorized text features as input variables
   - Target variable: star ratings (1-5)

3. **Train Linear Support Vector Classifier:**
   - Implement 5-fold cross-validation on training data
   - Optimize hyperparameters using validation set
   - Train final model on combined train+validation data

4. **Generate predictions:**
   - Apply trained model to test set
   - Generate predicted ratings
   - Save predictions alongside actual ratings

## Step 4: Model Evaluation
1. **Continue in `scripts/analysis.ipynb`**
   
2. **Calculate performance metrics:**
   - Overall accuracy (target: ≥80%)
   - Mean Absolute Error (MAE)
   - Precision, Recall, F1-score for each rating class
   - Generate confusion matrix

3. **Create visualizations:**
   - Predicted vs Actual ratings scatter plot
   - Confusion matrix heatmap
   - Review length analysis plots

## Step 5: Hypothesis Testing
1. **In `hypothesis_testing.ipynb`, prepare data for statistical testing:**
   - Calculate prediction errors (actual - predicted)
   - Classify predictions into three groups:
     - Over-predicted: Predicted > Actual
     - Under-predicted: Predicted < Actual  
     - Correctly predicted: Predicted = Actual

2. **Conduct hypothesis tests** (α = 0.05):

   **Test 1 - Over-Predicted Ratings:**
   - H₀: Mean difference (actual - predicted) ≥ 0
   - H₁: Mean difference < 0
   - Perform one-sample t-test

   **Test 2 - Under-Predicted Ratings:**
   - H₀: Mean difference ≤ 0  
   - H₁: Mean difference > 0
   - Perform one-sample t-test

   **Test 3 - Global Bias Test:**
   - H₀: Mean difference = 0 (no systematic bias)
   - H₁: Mean difference ≠ 0 (systematic bias exists)
   - Perform two-sided t-test

3. **Document results:**
   - Report test statistics, p-values, and conclusions
   - Interpret findings in context of model performance

## Step 6: Results Documentation
1. **Generate final outputs:**
   - All visualizations saved as PNG files
   - Statistical test results summary
   - Model performance metrics table

2. **Verify all outputs are created:**
   - `output/final_data.csv` - Cleaned dataset
   - Analysis results and visualizations in notebooks
   - Statistical test conclusions

## File Execution Order
1. `scripts/cleaning.ipynb` - Data preprocessing
2. `scripts/analysis.ipynb` - EDA, modeling
3. `scripts/hypothesis_testing.ipynb` - hypothesis testing

## Expected Outcomes
- Clean dataset with ~2.5M Amazon Fashion reviews
- Linear SVC model with ≥80% accuracy target
- Statistical evidence for/against systematic prediction bias
- Comprehensive analysis of model performance across rating categories
