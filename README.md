# ✈️ U.S. Domestic Flight Delay Analysis & Prediction

## 📌 Project Overview
This project was developed as part of the **AI 350: Data Science** course at **Jordan University of Science and Technology (JUST)**, under the supervision of **Dr. Malak Abdullah**.

The goal of this project is to apply data science and machine learning techniques to solve a real-world problem: **Predicting Flight Delays**. By analyzing historical flight data, this project aims to identify patterns and factors that contribute to delays, ultimately building models capable of predicting whether a flight will be delayed by more than 15 minutes.

## 🎯 Objectives
* Identify and resolve data quality issues including missing values, outliers, duplicates, and inconsistencies.
* Analyze delay patterns across airlines, airports, routes, and seasons.
* Develop and evaluate machine learning models to predict flight delay likelihood.

## 📊 Dataset
* **Source:** Bureau of Transportation Statistics (BTS) On-Time Reporting Carrier On-Time Performance database.
* **Description:** The dataset contains historical flight records from 2025 representing the winter, summer, and autumn seasons. A stratified random sample of 300,000 records was extracted to ensure data distribution integrity and computational efficiency.
* **Features:** Derived features engineered for analytical value include Airline_Name, Season, Route, and Scheduled_Hour.

## 🛠️ Methodology & Machine Learning Models
The project formulates the primary problem as a Binary Classification task (Is_Delayed: DEP_DELAY > 15 min). The following machine learning algorithms were implemented and compared:
1. **Logistic Regression**
2. **Random Forest**
3. **LightGBM**
4. **XGBoost**
5. **TabNet**
6. **Stacking Ensemble** (Meta-learner combining the predictions of base models)

*Additionally, regression models were evaluated to predict the exact delay duration, testing algorithms such as LightGBM, XGBoost, Random Forest, TabNet, and Linear Regression.*

## 📈 Key Results & Findings
After evaluating the models on the November Temporal Holdout dataset, we observed the following results for the classification task:
* **Best Performing Model:** The **Stacking Ensemble** achieved the highest performance with an AUC of **0.9295** and an F1-Score of **0.8070**.
* **Other Notable Models:** Logistic Regression achieved a highly competitive AUC of 0.9290, while Random Forest achieved an AUC of 0.9197.

For the regression task, **LightGBM** achieved the best performance with an R² score of **0.8775** and an MAE of 10.0988.

## 📁 Repository Structure
* `datascience05pre-eda.ipynb`: Jupyter Notebook containing data preprocessing and Exploratory Data Analysis (EDA).
* `datascience05model.ipynb`: Jupyter Notebook containing the implementation and evaluation of the machine learning models.
* `Research_paper.pdf`: The final research paper detailing the methodology, literature review, and comprehensive discussion of results.
* `Flight.pdf` / `Flight.docx`: Supporting project documentation outlining the research proposal.
* `.gitignore`: Specifies intentionally untracked files to ignore.

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/anasodat/Flight-Delay-Prediction.git](https://github.com/anasodat/Flight-Delay-Prediction.git) 
