import sqlite3
from datetime import datetime

DB_NAME = "alexa_reviews.db"

def init_db():
    """Creates the database and the reviews table if they do not exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            predicted_sentiment TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            timestamp DATETIME NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_review(text, sentiment, confidence):
    """Inserts a new analyzed review into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO reviews (review_text, predicted_sentiment, confidence_score, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (text, sentiment, confidence, current_time))
    
    conn.commit()
    conn.close()

def get_all_reviews():
    """Retrieves all rows from the reviews table ordered by timestamp."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM reviews ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database and table initialized successfully!")