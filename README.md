# 📊 Amazon Alexa Kullanıcı Yorumları - Duygu Analizi Paneli

Bu proje, Amazon Alexa cihazlarına yapılan kullanıcı yorumlarını işleyen, makine öğrenmesi modeliyle sınıflandıran (Duygu Analizi) ve sonuçları ilişkisel bir veritabanında saklayan uçtan uca (**End-to-End**) bir veri analitiği uygulamasıdır.

## 🚀 Özellikler (Features)
- **Gerçek Zamanlı NLP**: Kullanıcıdan alınan ham metin verileri anlık olarak işlenir.
- **Duygu Durumu Sınıflandırması**: Eğitilmiş optimum model ile yorumlar *Positive* veya *Negative* olarak ayrılır.
- **SQL Veritabanı Günlüğü**: Yapılan her tahmin, zaman damgası ve güven skoruyla SQLite3 veritabanına kaydedilir.
- **Dinamik Dashboard**: Güncel veri dağılım grafikleri arayüzde anlık olarak güncellenir.

## 🛠️ Teknoloji Yığıtı (Tech Stack)
- **Arayüz (Dashboard)**: Streamlit Framework
- **Makine Öğrenmesi (ML)**: Scikit-learn (TF-IDF Vectorizer & Classifier), Pickle
- **Veritabanı (Database)**: SQLite3 & Pandas
- **Görselleştirme**: Matplotlib & Seaborn

## 📦 Kurulum ve Çalıştırma (Setup)
1. Projeyi bilgisayarınıza indirin (Clone):
   ```bash
   git clone <REPOSİTORY_URL>