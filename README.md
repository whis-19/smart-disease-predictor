# 🧠 Smart Disease Predictor

Welcome to the **Smart Disease Predictor** — an AI-powered Streamlit app that classifies disease categories based on user-selected **symptoms, signs, and risk factors**.

> 🔬 Built with **TF-IDF**, **KNN**, and **Logistic Regression**, this tool showcases machine learning in a healthcare context.  
> ⚡ Explore interactive predictions and see model confidence in real-time!

---

## 🚀 Features

- ✅ Multi-select dropdowns for Symptoms, Signs, and Risk Factors
- ✅ Uses **TF-IDF encoding** for feature extraction
- ✅ K-Nearest Neighbors (KNN) classifier trained on real clinical patterns
- ✅ Probability-based pie chart for prediction confidence
- ✅ Clean, responsive UI powered by **Streamlit**

---

## 📊 Models Used

| Model             | Encoding         | Metrics Evaluated                     |
|-------------------|------------------|---------------------------------------|
| **KNN**           | TF-IDF           | Accuracy, Precision, Recall, F1-score |
| **Logistic Reg.** | TF-IDF & One-Hot | Accuracy, Precision, Recall, F1-score |

> 🔍 See full comparative analysis in the Jupyter Notebook.

---

## 🖥️ How to Run

```bash
streamlit run app.py
