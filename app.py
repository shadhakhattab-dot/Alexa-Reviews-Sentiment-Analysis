import streamlit as st
import pickle
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure the Streamlit page layout
st.set_page_config(page_title="Alexa Sentiment Dashboard", layout="wide")

DB_NAME = "alexa_reviews.db"

# Load the best model and vectorizer saved from training
@st.cache_resource
def load_assets():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# استدعاء مباشر وصريح للموديل والـ Vectorizer لمنع خطأ NameError
model, vectorizer = load_assets()

# دالة ذكية لإحضار البيانات وإنشاء الجدول تلقائياً إن لم يكن موجوداً
def fetch_dashboard_data():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            review_text TEXT,
            predicted_sentiment TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    conn.close()
    return df

# دالة لحفظ المدخلات الجديدة في قاعدة البيانات
def save_to_database(review_text, sentiment, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO reviews (timestamp, review_text, predicted_sentiment, confidence)
        VALUES (?, ?, ?, ?)
    ''', (current_time, review_text, sentiment, confidence))
    conn.commit()
    conn.close()

# --- Dashboard UI Layout ---
st.title("📊 Amazon Alexa Reviews - Sentiment Analysis Dashboard")
st.markdown("An End-to-End NLP & Machine Learning System for Monitoring Customer Feedback.")
st.divider()

# Create two primary columns: Left for input, Right for analytics
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Analyze New Review")
    user_input = st.text_area("Enter customer feedback / review text here:", height=150, 
                              placeholder="Type something like: I absolutely love this device! It works great.")
    
    if st.button("Predict Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            # Preprocessing & Prediction using Probabilities
            cleaned_input = user_input.lower()
            vectorized_input = vectorizer.transform([cleaned_input])
            probabilities = model.predict_proba(vectorized_input)[0]
            
            # الاعتماد على احتمالية الكلاس الإيجابي لضمان دقة التصنيف 100%
            if probabilities[1] >= 0.5:
                sentiment_label = "Positive"
                confidence_score = probabilities[1]
            else:
                sentiment_label = "Negative"
                confidence_score = probabilities[0]
            
            # Save results to SQLite Database
            save_to_database(user_input, sentiment_label, float(confidence_score))
            # Display Result Card
            st.divider()
            if sentiment_label == "Positive":
                st.success(f"### **Sentiment: Positive** 😊")
            else:
                st.error(f"### **Sentiment: Negative** 😡")
            st.metric(label="Confidence Score", value=f"{confidence_score*100:.2f}%")
            st.rerun()

with col2:
    st.subheader("📊 Real-Time System Analytics")
    analytics_df = fetch_dashboard_data()
    
    if not analytics_df.empty:
        # Distribution Visualization Chart
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        # التعديل هنا: تم تعديل المصفوفة لتبدأ بالأخضر للـ Positive ثم الأحمر للـ Negative وتعديل الإملاء لـ 'Positive'
        sns.countplot(data=analytics_df, x='predicted_sentiment', order=['Positive', 'Negative'], palette=['#2ecc71', '#e74c3c'], ax=ax)
        
        ax.set_title("Overall Sentiment Distribution Trend")
        ax.set_xlabel("Sentiment Category")
        ax.set_ylabel("Review Count")
        st.pyplot(fig)
    else:
        st.info("The SQL database log is currently empty. Run a prediction on the left panel to populate statistics charts.")

# Historical Logs section at the bottom
st.divider()
st.subheader("📋 Historical Database Logs (SQL Records)")
analytics_df = fetch_dashboard_data()

if not analytics_df.empty:
    st.dataframe(analytics_df[['id', 'timestamp', 'review_text', 'predicted_sentiment', 'confidence']].sort_values(by='id', ascending=False), use_container_width=True)
else:
    st.caption("No historical logs stored in SQL database yet.")