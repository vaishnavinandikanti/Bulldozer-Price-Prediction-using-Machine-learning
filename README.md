# Bulldozer-Price-Prediction-using-Machine-learning

Predicting the sale prices of heavy construction equipment using machine learning.

---

## 📌 Project Overview

This project aims to predict the **sale price of bulldozers** using historical data provided by auction listings. The dataset is from a Kaggle competition titled ["Blue Book for Bulldozers"](https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview), hosted by the auction company Ritchie Bros.

Using **Random Forest Regressor**, we built a model capable of accurately predicting sale prices based on a wide variety of features such as model type, usage hours, sale date, and more.

---

## 📊 Dataset

The dataset includes:

- **Training Set:** Bulldozer sales up to 2011.
- **Validation Set:** A held-out portion from the training set to evaluate model performance.
- **Test Set:** Sales after 2011, where the target values (`SalePrice`) are not provided (for prediction submission).

Key features:
- `MachineID`, `ModelID`, `YearMade`, `UsageBand`, `ProductSize`, `SaleDate`, etc.

---

## 🧠 ML Pipeline

### 🛠️ Preprocessing
- Parsing `saledate` into `year`, `month`, `day`, etc.
- Handling missing values with median for numeric and category encoding for object types.
- Creating indicator columns for missing values.

### 🔍 Feature Engineering
- Extracted temporal features from `saledate`.
- Encoded categorical variables with `pandas.Categorical().codes + 1`.

### ⚙️ Model Selection
- **Random Forest Regressor**
- Evaluation metric: **Root Mean Squared Log Error (RMSLE)**

### 🔎 Hyperparameter Tuning
Used `RandomizedSearchCV` to find optimal hyperparameters:
- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`, `min_samples_leaf`
- `max_samples`

---

## 🧪 Evaluation Metrics

| Metric        | Value (Train) | Value (Valid) |
|---------------|---------------|---------------|
| MAE           | 2953.81       | 5951.25       |
| RMSLE         | 0.1447        | 0.2452        |
| R² Score      | 0.9588        | 0.8818        |

---

## 🔮 Predictions

- The best model was used to generate predictions on the test dataset.
- Preprocessed the test set to match training features, including any missing indicator columns.
- Predictions exported as `test_predictions.csv`.

---

## 📈 Feature Importance

Visualized the top 20 features contributing to predictions. Key influencers included:
- `YearMade`
- `ProductSize`
- `saleYear`
- `ModelID`

## 📊 Model Evaluation

The model is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Log Error (RMSLE)**
- **R² Score**

These metrics help assess how well the model generalizes to unseen data and its overall prediction accuracy.

---


