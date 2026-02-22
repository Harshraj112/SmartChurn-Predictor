# 🔄 SmartChurn Predictor

A deep learning-powered customer churn prediction system built with **TensorFlow/Keras** and deployed as an interactive web application using **Streamlit**. The model uses an Artificial Neural Network (ANN) trained on real-world banking customer data to predict the likelihood of a customer churning.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Results](#results)
- [Author](#author)

---

## 🧠 Overview

Customer churn — when a customer stops using a company's service — is one of the most costly problems for businesses. SmartChurn Predictor helps businesses identify **at-risk customers** before they leave, allowing for proactive retention strategies.

The app accepts customer details as input (geography, age, balance, credit score, etc.) and outputs:

- **Churn Probability** (0.0 to 1.0)
- A clear classification: _"Likely to churn"_ or _"Not likely to churn"_

---

## 🚀 Demo

Run the app locally:

```bash
streamlit run app.py
```

---

## ✨ Features

- 🔮 **Real-time Churn Prediction** — Instant probability scores via an ANN
- 🧪 **Hyperparameter Tuning** — Grid search over ANN configurations using KerasTuner / SciKeras
- 📊 **Salary Regression** — Auxiliary regression notebook predicting estimated salary
- 📈 **TensorBoard Logs** — Training metrics visualized with TensorBoard
- 🔄 **Preprocessing Pipeline** — Label encoding, one-hot encoding, and standard scaling persisted as `.pkl` files
- 🖥️ **Streamlit UI** — Clean, interactive web interface with sliders, dropdowns, and number inputs

---

## 📁 Project Structure

```
SmartChurn Predictor/
│
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Main EDA, model training & evaluation
├── hyperparametertuningann.ipynb   # ANN hyperparameter tuning experiments
├── prediction.ipynb                # Standalone prediction notebook
├── salaryregression.ipynb          # Regression model for salary prediction
│
├── model.h5                        # Trained Keras ANN model
├── label_encoder_gender.pkl        # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl          # Fitted OneHotEncoder for Geography
├── scaler.pkl                      # Fitted StandardScaler
│
├── Churn_Modelling.csv             # Source dataset
├── requirements.txt                # Python dependencies
├── logs/                           # TensorBoard training logs
│   └── fit/
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category               | Tools                   |
| ---------------------- | ----------------------- |
| **Language**           | Python 3.x              |
| **Deep Learning**      | TensorFlow 2.15, Keras  |
| **ML / Preprocessing** | Scikit-learn, SciKeras  |
| **Data Handling**      | Pandas, NumPy           |
| **Visualization**      | Matplotlib, TensorBoard |
| **Web App**            | Streamlit               |
| **Serialization**      | Pickle                  |

---

## 📊 Dataset

**Source:** [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling)

| Feature           | Description                                            |
| ----------------- | ------------------------------------------------------ |
| `CreditScore`     | Customer's credit score                                |
| `Geography`       | Country (France, Germany, Spain)                       |
| `Gender`          | Male / Female                                          |
| `Age`             | Customer age                                           |
| `Tenure`          | Years with the bank                                    |
| `Balance`         | Account balance                                        |
| `NumOfProducts`   | Number of bank products used                           |
| `HasCrCard`       | Has a credit card (0 / 1)                              |
| `IsActiveMember`  | Active bank member (0 / 1)                             |
| `EstimatedSalary` | Estimated annual salary                                |
| `Exited`          | **Target** — Did the customer churn? (1 = Yes, 0 = No) |

- **Total Records:** ~10,000
- **Target Distribution:** ~20% churn (imbalanced)

---

## 🏗️ Model Architecture

The core model is a fully connected **Artificial Neural Network (ANN)**:

```
Input Layer   →  12 features (after encoding)
Hidden Layer 1 → 64 neurons, ReLU activation
Hidden Layer 2 → 32 neurons, ReLU activation
Output Layer  →  1 neuron, Sigmoid activation
```

- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Callbacks:** EarlyStopping, TensorBoard

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Harshraj112/SmartChurn-Predictor.git
cd SmartChurn-Predictor
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

### Input the following customer details:

- Geography, Gender, Age, Tenure
- Balance, Credit Score, Estimated Salary
- Number of Products, Has Credit Card, Is Active Member

The app will instantly display the **churn probability** and a prediction verdict.

### View TensorBoard Logs

```bash
tensorboard --logdir logs/fit
```

---

## 📓 Notebooks

| Notebook                        | Description                                                      |
| ------------------------------- | ---------------------------------------------------------------- |
| `experiments.ipynb`             | Full pipeline: EDA → preprocessing → model training → evaluation |
| `hyperparametertuningann.ipynb` | Grid/random search for optimal ANN hyperparameters               |
| `prediction.ipynb`              | Load the saved model and run predictions on new data             |
| `salaryregression.ipynb`        | Regression model to predict `EstimatedSalary` using ANN          |

---

## 📈 Results

| Metric                | Value                         |
| --------------------- | ----------------------------- |
| **Accuracy**          | ~86%                          |
| **Model Type**        | Binary Classification ANN     |
| **Prediction Output** | Churn Probability (0.0 – 1.0) |

> Threshold: A probability **> 0.5** classifies the customer as likely to churn.

---

## 👤 Author

**Harsh Raj**

- GitHub: [@Harshraj112](https://github.com/Harshraj112)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
