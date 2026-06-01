# Car Price Prediction

A machine learning project to predict car prices based on key vehicle attributes using regression models.

---

## Project Overview

This project builds an end-to-end ML regression pipeline to predict car prices based on vehicle attributes such as make, model, year, engine size, mileage, fuel type, and transmission. The dataset contains 1000 records across 5 car brands — Honda, Ford, BMW, Audi, and Toyota — with prices as the continuous target variable.
The pipeline covers data preprocessing, categorical encoding, outlier treatment, feature scaling, and selection, followed by training and comparison of 7 regression models evaluated using R² score.

**Brands covered:**
- Honda
- Ford
- BMW
- Audi
- Toyota

---

## Repository Structure

car-price-prediction  
 ┣ Car_price_prediction.ipynb — Full ML pipeline notebook  
 ┗ Car_Price_Prediction.csv — Dataset (1000 records)

---

## Dataset

| Feature | Description |
|---|---|
| `Make` | Car brand |
| `Model` | Car model |
| `Year` | Manufacturing year |
| `Engine Size` | Engine size in litres |
| `Mileage` | Total mileage driven |
| `Fuel Type` | Petrol, Diesel, or Electric |
| `Transmission` | Manual or Automatic |
| `Price` | Target variable — car price |

- **Total records:** 1000
- **Brands:** Honda, Ford, BMW, Audi, Toyota
- **Fuel types:** Petrol, Diesel, Electric
- **Transmission:** Manual, Automatic

---

## ML Pipeline

### 1. Data Preprocessing
- Null value imputation (mean for numerical, mode for categorical)
- One-Hot Encoding for `Make`, `Model`, `Fuel Type`, `Transmission`
- IQR-based outlier treatment
- StandardScaler feature normalization

### 2. Feature Selection
- `SelectKBest` with F-regression (`f_regression`)
- Top 10 features selected: Year, Engine Size, Mileage, and encoded car attributes

### 3. Train-Test Split
- 70% training / 30% testing (`random_state=50`)

### 4. Models Trained

| Model | Library |
|---|---|
| Linear Regression | `sklearn.linear_model` |
| Decision Tree | `sklearn.tree` |
| Random Forest | `sklearn.ensemble` |
| K-Nearest Neighbors | `sklearn.neighbors` |
| Support Vector Regression | `sklearn.svm` |
| Gradient Boosting | `sklearn.ensemble` |
| Ridge Regression | `sklearn.linear_model` |

- **Evaluation Metric:** R² Score

---

## Author

**Jaishri Vijayakumar**  
B.Sc. Data Science | PSGR Krishnammal College for Women, Coimbatore  
LinkedIn: linkedin.com/in/jaishri-vijayakumar  
GitHub: github.com/jaishrivijayakumar
