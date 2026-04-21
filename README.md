# ❤️ Heart Disease Prediction System

## 📌 Overview

This project is an end-to-end **Machine Learning application** that predicts the likelihood of heart disease using patient medical data.
It includes data preprocessing, model training, evaluation, and an interactive **Streamlit dashboard**.

---

## 🚀 Features

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Random Forest ML model
* Feature importance visualization
* Interactive dashboard with real-time prediction

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Streamlit
* Joblib

---

## 📂 Project Structure

```
heart_project.py
app.py
model.pkl
scaler.pkl
cleaned_data.csv
images/
README.md
```

---

## 📊 Dataset

* Source: UCI Machine Learning Repository
* Domain: Healthcare
* Task: Heart Disease Prediction

  Data set (sample) :
  ![dataset](images/dataset.png)

---

# 📸 APPLICATION PREVIEW

## 🖥️ Main Dashboard

# 🌐 Dashboard Features

* Interactive sidebar inputs
* Real-time prediction
* Data visualization charts
* Feature importance display
* Dataset preview option


![dashboard1](images/dashboard.png)

![dashboard2](images/dashboard.png)

---

## 🎛️ Sidebar Input Panel

Users can input patient details such as age, cholesterol, blood pressure, etc.

![sidebar](images/sidebar.png)

---

## 🔍 Prediction Output

Displays whether the patient is at **high risk** or **low risk**.

![predictionbutton](images/prediction.png)

---

# 📈 DATA ANALYSIS (EDA)

## 📊 Target Distribution

![Target Distribution](images/target_distribution.png)

---

## 🔥 Correlation Heatmap

![Correlation](images/correlation.png)

---

## 📉 Feature Importance

![Feature Importance](images/feature_importance.png)

---

# ⚙️ Setup Instructions

## 1️⃣ Clone Repository

```
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

## 2️⃣ Install Dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib ucimlrepo
```

---

# ▶️ How to Run

## Step 1: Train Model

```
python heart_project.py
```

## Step 2: Run App

```
streamlit run app.py
```
Sample Output  : 

Input : 

![sampleinput](images/sampleinput.png)

Output : 

![sampleoutput](images/sampleoutput.png)

---

# 🤖 Model Details

* Algorithm: Random Forest Classifier
* Preprocessing: StandardScaler
* Train/Test Split: 80/20

---

# 💡 Key Insights

* Age and cholesterol significantly impact predictions
* Chest pain type is a strong indicator
* Model helps in early detection of heart disease

---

# 🎯 Conclusion

This project demonstrates a complete **machine learning pipeline** and its deployment using an interactive dashboard for healthcare prediction.

---

# 🙌 Acknowledgment

Dataset provided by UCI Machine Learning Repository.

---
