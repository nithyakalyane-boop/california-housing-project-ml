# 🏠 California Housing Price Prediction (End-to-End ML Project)

A production-style machine learning pipeline to predict California housing prices using regression models (XGBoost, Random Forest) with preprocessing, feature engineering, and optional hyperparameter tuning.

---

## 🚀 Project Overview

This project demonstrates a **complete ML workflow**:

* Data ingestion & cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Preprocessing (encoding + scaling)
* Model training (Random Forest / XGBoost)
* Hyperparameter tuning (Optuna)
* Model evaluation
* One-command execution pipeline

---

## 📁 Project Structure

```
california-housing-ml/
│
├── data/
│   ├── raw/
│   │   └── housing.csv
│   ├── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── model.py
│   ├── optuna_tuning.py
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/nithyakalyane/california-housing-ml.git
cd california-housing-ml
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` not ready yet, install manually:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn optuna
```

---

## ▶️ Run the Project (ONE COMMAND)

### Train model

```bash
python main.py --mode train
```

### Run EDA

```bash
python main.py --mode eda
```

### Run full pipeline

```bash
python main.py --mode all
```

---

## 📊 Model Used

* Random Forest Regressor
* XGBoost Regressor (recommended)
* Optuna for hyperparameter tuning

---

## 📈 Features

* Median Income
* House Age
* Average Rooms
* Population
* Latitude / Longitude
* Ocean proximity encoding

---

## 🧠 Skills Demonstrated

* End-to-end ML pipeline design
* Data preprocessing & feature engineering
* Model comparison & evaluation
* Hyperparameter tuning (Optuna)
* Modular Python project structure
* GitHub-ready ML project

---

## 🧪 Expected Output

* RMSE score printed
* Trained model saved (`model.pkl`)
* Clean evaluation metrics

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
optuna
```

---

## 📌 Next Improvements

* Add FastAPI deployment
* Dockerize project
* CI/CD with GitHub Actions
* Streamlit dashboard

---

## 👨‍💻 Author

Built as a portfolio-level ML project for showcasing production-ready skills.

