# Fantasy Premier League: Who Deserves More Points?

An end-to-end data science project using live data from the official Fantasy Premier League (FPL) API to identify undervalued, high-performing players whose actual points lag behind their underlying statistics.

---

## 📌 Project Overview

In Fantasy Premier League, managers operate on a fixed budget, and price changes happen quickly once a player hits a streak of form. Often, a player's raw point total is distorted by variance—hit woodwork, missed sitters, or an opposing goalkeeper having an extraordinary match.

**The Goal:** Train a machine learning model to predict a player's *expected* FPL points based on underlying performance metrics (minutes played, starts, expected goals, expected assists, expected goals conceded, and defensive contributions), then compare those predictions against their actual points to uncover "unlucky" buy-low targets before their returns regress to the mean.

---

## 🗂️ Workflow & Methodology

The notebook follows a structured 9-step data science pipeline:

1. **Problem Framing:** Defining the FPL value-identification problem and target metrics.
2. **Data Ingestion:** Fetching live JSON data directly from the FPL `bootstrap-static` API endpoint (`elements`, `teams`, and `element_types`).
3. **Data Inspection:** Auditing 670+ player records across 109 features and identifying data type mismatches.
4. **Data Cleaning & Filtering:**
   - Mapping numeric IDs to readable team names and positions (`GKP`, `DEF`, `MID`, `FWD`).
   - Converting `now_cost` to actual FPL prices (£m) and casting string-encoded metrics (`expected_goals`, `expected_assists`, etc.) to numeric types.
   - Filtering out low-sample noise by restricting the dataset to players with **at least 270 minutes** played (3 full matches).
5. **Exploratory Data Analysis (EDA):**
   - Visualizing top overall point scorers by position.
   - Analyzing the relationship between player price and total points to locate budget value.
   - Plotting Expected Goals ($xG$) against actual goals scored to spot finishing over- and under-performers.
6. **Model Building:** Training regression models (`LinearRegression` and `RandomForestRegressor`) using `scikit-learn` to estimate fair point totals.
7. **Model Evaluation:** Assessing predictive accuracy using cross-validation, Mean Absolute Error ($MAE$), and $R^2$ scores.
8. **Player Scouting (Inference):** Calculating the residual difference ($\text{Predicted Points} - \text{Actual Points}$) to rank the most undervalued assets in the league.
9. **Reporting:** Summarizing actionable transfer recommendations.

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **Data Manipulation & API Requests:** `requests`, `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `train_test_split`, `cross_val_predict`)

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed along with the required libraries:

```bash
pip install -r requirements.txt
