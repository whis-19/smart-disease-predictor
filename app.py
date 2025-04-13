import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack

# ---------- Custom Streamlit Page Config ----------
st.set_page_config(page_title="AI Disease Predictor", page_icon="🧬", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2718/2718224.png", width=100)
    st.title("🩺 Smart Diagnosis")
    st.markdown("""
        This app uses **AI (KNN + TF-IDF)** to predict a disease category based on selected symptoms, signs, and risk factors.

        👉 Select from dropdowns  
        🎯 Click **Predict**  
        📊 View confidence chart  
    """)
    st.info("Built with 💙 using Streamlit, scikit-learn, and Python")

# ---------- Load and Process Data ----------
@st.cache_data
def load_data():
    df_raw = pd.read_csv("disease_features.csv")
    df = df_raw.copy()
    for col in ['Risk Factors', 'Symptoms', 'Signs']:
        df[col] = df[col].apply(lambda x: " ".join(ast.literal_eval(x)))
    return df_raw, df

df_raw, df = load_data()

# ---------- Extract Keywords ----------
import re

def extract_keywords(column):
    phrases = set()
    for item in df_raw[column]:
        try:
            parsed = ast.literal_eval(item)
            if isinstance(parsed, list):
                for term in parsed:
                    cleaned = re.sub(r'\s+', ' ', str(term)).strip().lower()
                    phrases.add(cleaned)
        except Exception:
            continue
    return sorted(phrases)

risk_options = extract_keywords("Risk Factors")
symptom_options = extract_keywords("Symptoms")
sign_options = extract_keywords("Signs")

# ---------- TF-IDF Vectorization ----------
tfidf_risk = TfidfVectorizer()
tfidf_symptoms = TfidfVectorizer()
tfidf_signs = TfidfVectorizer()

X_risk = tfidf_risk.fit_transform(df['Risk Factors'])
X_symptoms = tfidf_symptoms.fit_transform(df['Symptoms'])
X_signs = tfidf_signs.fit_transform(df['Signs'])
X_tfidf = hstack([X_risk, X_symptoms, X_signs])

# ---------- Target Labels ----------
category_map = {
    "Acute Coronary Syndrome": "Cardiovascular",
    "Aortic Dissection": "Cardiovascular",
    "Atrial Fibrillation": "Cardiovascular",
    "Heart Failure": "Cardiovascular",
    "Hypertensive Emergency": "Cardiovascular",
    "Myocardial Infarction": "Cardiovascular",
    "Asthma": "Respiratory",
    "COPD": "Respiratory",
    "Pneumonia": "Respiratory",
    "Alzheimer": "Neurological",
    "Migraine": "Neurological",
    "Seizure": "Neurological",
    "Stroke": "Neurological",
    "Hypoglycemia": "Metabolic",
    "Type I Diabetes": "Metabolic",
    "Type II Diabetes": "Metabolic",
    "Hyperthyroidism": "Endocrine",
    "Hypothyroidism": "Endocrine",
    "Adrenal Insufficiency": "Endocrine",
    "Appendicitis": "Gastrointestinal",
    "Gastritis": "Gastrointestinal",
    "Peptic Ulcer": "Gastrointestinal"
}
df['Category'] = df['Disease'].map(category_map).fillna("Other")
y = df['Category']

# ---------- Train Model ----------
model = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
model.fit(X_tfidf, y)

# ---------- Main App UI ----------
st.markdown("# 🤖 AI Disease Category Predictor")
st.markdown("### Select your known symptoms, signs, and risk factors below to get a prediction.")

# Two column layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    selected_risks = st.multiselect("🧪 Risk Factors", options=risk_options)

with col2:
    selected_symptoms = st.multiselect("🤒 Symptoms", options=symptom_options)

with col3:
    selected_signs = st.multiselect("🔍 Signs", options=sign_options)

st.markdown("---")

# ---------- Prediction Logic ----------
if st.button("🚀 Predict Disease Category"):
    risk_input = " ".join(selected_risks)
    symptom_input = " ".join(selected_symptoms)
    sign_input = " ".join(selected_signs)

    risk_vec = tfidf_risk.transform([risk_input])
    symptom_vec = tfidf_symptoms.transform([symptom_input])
    sign_vec = tfidf_signs.transform([sign_input])
    combined_vec = hstack([risk_vec, symptom_vec, sign_vec])

    prediction = model.predict(combined_vec)[0]
    proba = model.named_steps['kneighborsclassifier'].predict_proba(combined_vec)[0]
    labels = model.named_steps['kneighborsclassifier'].classes_

    confidence = round(proba[labels.tolist().index(prediction)] * 100, 2)

    # Display result
    st.success(f"🧠 Predicted Category: **{prediction}**")
    st.info(f"📈 Confidence: **{confidence}%**")

    # Pie chart
    fig, ax = plt.subplots()
    colors = cm.Paired(range(len(labels)))
    ax.pie(proba, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    st.markdown("### 🔬 Prediction Probabilities")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Created by Muhammad Abdullah — Powered by Machine Learning & Streamlit")
