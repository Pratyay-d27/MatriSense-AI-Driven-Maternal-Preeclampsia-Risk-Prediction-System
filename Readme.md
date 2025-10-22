<h1 align="center">🩺 MatriSense</h1>
<h3 align="center">AI-Powered Maternal & Preeclampsia Risk Prediction System</h3>

<p align="center">
  <em>Empowering healthcare professionals with intelligent diagnostics for early detection and prevention.</em>
</p>

---

## 🚀 Overview

The domain of **maternal health** is an area less explored, which inspired the development of **MatriSense** — an AI-powered early detection system aimed at predicting maternal health risks and identifying the likelihood of **preeclampsia onset**.  

**MatriSense** first evaluates **maternal health risks** using a **Random Forest model with LDA-based dimensionality reduction**. If the predicted risk is **medium or high**, the system automatically proceeds to predict **preeclampsia** using a **3-layer ANN**. This workflow ensures that early warning is provided only when necessary, optimizing attention for high-risk cases.

Built with **Streamlit**, **Scikit-Learn**, and **TensorFlow**, the system also integrates **LIME** for **explainable AI**, ensuring transparency and clinical trust.

---

## 🌟 Features

- 🧠 **Dual AI Models** — Predicts both maternal health risk and preeclampsia.
- 🔍 **Explainable AI** — Provides insights using **LIME**.
- 💅 **Elegant Streamlit UI** — Modern gradient design with interactive user inputs.
- 📊 **Real-time Prediction** — Instant evaluation of user-provided features with visual progress bars.
- 🛡️ **Clinically Focused** — Tailored for health analytics, research, and early intervention.

---

## 🧩 Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| Frontend | Streamlit, HTML/CSS |
| Backend | Python |
| Machine Learning | Scikit-Learn, TensorFlow |
| Explainability | LIME |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |

---

## 🩸 Model Architecture

### **Maternal Health Risk Model**
- **Algorithm:** Random Forest Classifier  
- **Dimensionality Reduction:** Linear Discriminant Analysis (LDA)  
- **Explainability:** LIME (Local Interpretable Model-Agnostic Explanations)  

### **Preeclampsia Prediction Model**
- **Architecture:** 3-layer Artificial Neural Network (ANN)  
- **Activation Functions:** ReLU, Sigmoid    

---

## 🧭 How it Works

1. **User Inputs:** Age, blood pressure, blood sugar, body temperature, and heart rate.  
2. **Maternal Risk Evaluation:** Random Forest + LDA predicts **low, medium, or high risk**.  
3. **Conditional Preeclampsia Prediction:** If risk is medium/high, ANN predicts likelihood of **preeclampsia**.  
4. **Explainable Insights:** LIME explains maternal risk.  
5. **Result Visualization:** Risk levels are displayed with color-coded progress bars and clear messages.

---

## 🧭 How to Run Locally

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/MatriSense.git
cd MatriSense
