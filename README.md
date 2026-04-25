# 🚀End-to-End-ML-Web-App

A complete end-to-end **Machine Learning Web Application** built using **Python, Flask, and Scikit-learn** that allows users to upload datasets, train models, compare performance, and make predictions — all from a simple web interface.

---

## 📌 Features

### 📂 1. Dataset Upload

* Upload CSV or Excel files
* Preview dataset (first 10 rows)
* View dataset columns for target selection

### 🧹 2. Data Preprocessing

* Remove duplicate rows
* Drop useless columns (constant / ID columns)
* Handle missing values
* Encode categorical variables
* Feature scaling using StandardScaler

### 🤖 3. Model Training

Supports both:

#### ✅ Classification Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

#### 📈 Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

---

### 📊 4. Model Evaluation

* Automatically selects best model
* Displays model performance (Accuracy / R²)
* Use Accuracy for classification & R² for Regression

---

### 🔮 5. Prediction Module

* Dynamic input form based on dataset features
* Handles categorical encoding during prediction
* Applies same preprocessing (scaling + encoding)
* Returns prediction result

---

### 🎨 6. UI Features

* Clean and responsive design
* Step-by-step workflow:

  ```
  Upload → Preview → Preprocess → Train → Predict
  ```

---

## 🏗️ Project Structure

```
ml_web_app_pro/
│
├── app.py                  # Main Flask application
├── requirements.txt       # Dependencies
│
├── modules/
│   ├── preprocess.py      # Data cleaning & preprocessing
│   ├── train.py           # Model training
│   ├── predict.py         # Prediction logic
│
├── templates/
│   ├── index.html
│   ├── preview.html
│   ├── train.html
│   ├── predict.html
│   ├── result.html
│
├── static/
│   ├── css/style.css
│
├── uploads/               # Uploaded datasets
├── models/                # Saved models (future use)
```

---



## 🧠 How It Works

1. User uploads dataset
2. Selects target column
3. App preprocesses data automatically
4. Multiple ML models are trained
5. Best model is selected
6. User inputs new data
7. Model returns prediction

---

## 💡 Tech Stack

* **Backend:** Flask (Python)
* **ML:** Scikit-learn, Pandas, NumPy
* **Frontend:** HTML, CSS
* **Storage:** Local file system

---
