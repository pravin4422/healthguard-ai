import sqlite3
import pandas as pd
from datetime import datetime

def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS diabetes_predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  pregnancies INTEGER,
                  glucose REAL,
                  blood_pressure REAL,
                  skin_thickness REAL,
                  insulin REAL,
                  bmi REAL,
                  dpf REAL,
                  age INTEGER,
                  prediction INTEGER,
                  probability REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS disease_predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  age INTEGER,
                  gender TEXT,
                  fever TEXT,
                  cough TEXT,
                  fatigue TEXT,
                  difficulty_breathing TEXT,
                  blood_pressure TEXT,
                  cholesterol TEXT,
                  disease TEXT,
                  confidence REAL)''')
    conn.commit()
    conn.close()

def save_diabetes_prediction(data, prediction, probability):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO diabetes_predictions 
                 (timestamp, pregnancies, glucose, blood_pressure, skin_thickness, 
                  insulin, bmi, dpf, age, prediction, probability)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data['Pregnancies'], 
               data['Glucose'], data['BloodPressure'], data['SkinThickness'],
               data['Insulin'], data['BMI'], data['DiabetesPedigreeFunction'], 
               data['Age'], prediction, probability))
    conn.commit()
    conn.close()

def save_disease_prediction(age, gender, fever, cough, fatigue, difficulty_breathing, 
                            blood_pressure, cholesterol, disease, confidence):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO disease_predictions 
                 (timestamp, age, gender, fever, cough, fatigue, difficulty_breathing,
                  blood_pressure, cholesterol, disease, confidence)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), age, gender, fever, 
               cough, fatigue, difficulty_breathing, blood_pressure, cholesterol, 
               disease, confidence))
    conn.commit()
    conn.close()

def get_diabetes_history():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query('SELECT * FROM diabetes_predictions ORDER BY timestamp DESC', conn)
    conn.close()
    return df

def get_disease_history():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query('SELECT * FROM disease_predictions ORDER BY timestamp DESC', conn)
    conn.close()
    return df
