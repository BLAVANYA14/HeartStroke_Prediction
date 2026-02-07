# 🫀 Heart Stroke Prediction using Machine Learning

## 📌 Project Overview

This project is a **Heart Stroke Prediction System** built using **Machine Learning** to predict whether a person is at risk of having a stroke based on medical and lifestyle attributes.

The model analyzes patient health data such as age, hypertension, heart disease, glucose level, BMI, and smoking status to classify the likelihood of a **stroke (Yes / No)**. This kind of system can assist healthcare professionals and individuals in **early risk assessment and preventive care**.

---

## 🛠️ Technologies & Libraries Used

* **Python**
* **Pandas & NumPy** – Data handling and numerical computation
* **Scikit-learn** – Model training, preprocessing, evaluation
* **Matplotlib / Seaborn** (optional) – Data visualization

---

## 📂 Dataset Description

* Dataset file: `healthcare-dataset-stroke-data.csv`
* Each row represents a patient record

### Features Used

* `gender`
* `age`
* `hypertension`
* `heart_disease`
* `ever_married`
* `work_type`
* `Residence_type`
* `avg_glucose_level`
* `bmi`
* `smoking_status`

### Target Variable

* `stroke`

  * `0` → No Stroke
  * `1` → Stroke

---

## 🧹 Data Preprocessing

The following preprocessing steps are applied:

1. Removed unnecessary columns (e.g., `id`)
2. Handled missing values:

   * `bmi` filled with **median value**
3. Converted categorical variables using **Label Encoding / One-Hot Encoding**
4. Feature scaling (if required)
5. Split data into **training and testing sets**

These steps ensure clean, consistent, and model-ready data.

---

## 🤖 Machine Learning Model

* **Algorithm Used**: Logistic Regression / Random Forest (depending on implementation)
* **Train-Test Split**: 80% training, 20% testing

### Why this model?

* Works well for binary classification
* Easy to interpret
* Handles numerical and categorical health data effectively

---

## 📊 Model Evaluation

The model is evaluated using:

* **Accuracy Score**
* **ROC-AUC Score**
* **Classification Report**

  * Precision
  * Recall
  * F1-score

### Sample Results

* **Accuracy**: ~92%
* **ROC-AUC Score**: ~0.81

Although the dataset is imbalanced, ROC-AUC gives a better picture of the model’s performance.

---

## ⚠️ Class Imbalance Note

Stroke cases are significantly fewer compared to non-stroke cases. This may lead to:

* High overall accuracy
* Lower recall for stroke-positive cases

Future improvements can address this using **SMOTE, class weighting, or advanced models**.

---

## ▶️ How to Run the Project

1. Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Place the dataset in the project folder

3. Run the script:

```bash
python stroke_prediction.py
```

4. View model accuracy, ROC-AUC score, and classification report in the console

---

## 🔮 Future Enhancements

* Handle class imbalance using **SMOTE**
* Try advanced models like **XGBoost, LightGBM**
* Hyperparameter tuning
* Build a **Flask / Streamlit web app**
* Save model using `joblib` for deployment

---

## 📌 Conclusion

This project demonstrates a complete **end-to-end Machine Learning pipeline** for predicting stroke risk, including data preprocessing, feature engineering, model training, evaluation, and result interpretation. It highlights how ML can support **healthcare decision-making** through data-driven insights.

---

✨ *Built with Machine Learning for Healthcare Applications*
