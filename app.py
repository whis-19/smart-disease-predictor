import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import load_npz, hstack
import joblib

# Load datasets and models
@st.cache_data
def load_data():
    df = pd.read_csv("encoded_output2_with_categories.csv")
    one_hot = df.drop(columns=["Disease", "Category"]).values
    tfidf = load_npz("tfidf_combined_matrix.npz")
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["Category"])
    return df, one_hot, tfidf, labels, label_encoder

@st.cache_resource
def load_vectorizers():
    return {
        "risk": joblib.load("vectorizer_risk.pkl"),
        "symptoms": joblib.load("vectorizer_symptoms.pkl"),
        "signs": joblib.load("vectorizer_signs.pkl"),
        "subtypes": joblib.load("vectorizer_subtypes.pkl"),
    }

df, one_hot_matrix, tfidf_matrix, labels, label_encoder = load_data()
vectorizers = load_vectorizers()

# UI - Header
st.title("🧠 Disease Category Classifier (KNN)")
st.markdown("This app uses **K-Nearest Neighbors** to classify diseases based on symptoms, risk factors, signs, and subtypes.")

# Sidebar - Config
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("Use the options below to configure the classifier.")
encoding = st.sidebar.selectbox("Encoding Method", ["TF-IDF", "One-Hot"])
k = st.sidebar.selectbox("Number of Neighbors (k)", [3, 5, 7])
metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"])

# Input area
with st.expander("✏️ Input Details for Prediction", expanded=True):
    st.markdown("Provide the details below to make a prediction.")
    risk_input = st.text_input("Risk Factors", "fever stress")
    symptoms_input = st.text_input("Symptoms", "chills cough")
    signs_input = st.text_input("Signs", "wheezing sneezing")
    subtypes_input = st.text_input("Subtypes", "viral")  # optional

# TF-IDF transformation function
def transform_input_tfidf(risk, symptoms, signs, subtypes):
    r = vectorizers["risk"].transform([risk])
    s = vectorizers["symptoms"].transform([symptoms])
    si = vectorizers["signs"].transform([signs])
    su = vectorizers["subtypes"].transform([subtypes])
    return hstack([r, s, si, su])

# One-Hot Encoding function (dynamic from training set structure)
def transform_input_one_hot(risk, symptoms, signs, subtypes):
    all_columns = df.drop(columns=["Disease", "Category"]).columns
    input_vector = np.zeros(len(all_columns))

    for i, col in enumerate(all_columns):
        # check if the keyword (e.g., 'fever') is in user input
        if (
            any(word in col.lower() for word in risk.lower().split())
            or any(word in col.lower() for word in symptoms.lower().split())
            or any(word in col.lower() for word in signs.lower().split())
            or any(word in col.lower() for word in subtypes.lower().split())
        ):
            input_vector[i] = 1
    return input_vector

# Predict & Evaluate
if st.button("🔍 Predict & Evaluate"):
    with st.spinner("Processing..."):
        # Select matrix based on encoding choice
        X = tfidf_matrix if encoding == "TF-IDF" else one_hot_matrix

        # Create input vector
        if encoding == "TF-IDF":
            input_vector = transform_input_tfidf(risk_input, symptoms_input, signs_input, subtypes_input)
        elif encoding == "One-Hot":
            input_vector = transform_input_one_hot(risk_input, symptoms_input, signs_input, subtypes_input)
        else:
            st.warning("Please select a valid encoding method.")
            input_vector = None

        # Train and predict
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X, labels)

        if input_vector is not None:
            # Reshape if needed
            if encoding == "TF-IDF":
                prediction = model.predict(input_vector)[0]
            else:
                prediction = model.predict([input_vector])[0]

            predicted_label = label_encoder.inverse_transform([prediction])[0]
            st.success(f"🧾 Predicted Category: {predicted_label}")

        # Evaluate with 5-fold cross-validation
        st.subheader("📊 Cross-Validation Metrics (5-Fold)")
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        results = cross_validate(model, X, labels, scoring=scoring, cv=StratifiedKFold(n_splits=5))

        st.write(f"**Accuracy:** {np.mean(results['test_accuracy']):.3f}")
        st.write(f"**Precision:** {np.mean(results['test_precision_macro']):.3f}")
        st.write(f"**Recall:** {np.mean(results['test_recall_macro']):.3f}")
        st.write(f"**F1 Score:** {np.mean(results['test_f1_macro']):.3f}")

# Show dataset sample
with st.expander("🧾 View Dataset Sample"):
    st.dataframe(df.head())
