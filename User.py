#Author : t.me/Jelofen1962
#MIT Copy-Right, you can use this model but just With the CREDIT

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from numpy import loadtxt

def get_user_input():
    height = float(input("Enter your height (cm): "))
    weight = float(input("Enter your weight (kg): "))
    age = int(input("Enter your age: "))
    kids = int(input("Enter number of kids: "))
    gender = int(input("Enter your gender (0 for male, 1 for female): "))
    employment_status = int(input("Are you employed? (0 for no, 1 for yes): "))
    marital_status = int(input("Are you married? (0 for no, 1 for yes): "))
    bmi = weight / ((height / 100) ** 2)
    smoking = int(input("Do you smoke? (0 for no, 1 for yes): "))
    alcohol = int(input("Do you drink alcohol? (0 for no, 1 for yes): "))
    exercise = int(input("Do you exercise regularly? (0 for no, 1 for yes): "))
    diet = int(input("Do you follow a healthy diet? (0 for no, 1 for yes): "))
    stress_level = int(input("Rate your stress level (1-10): "))
    sleep_hours = float(input("Average sleep hours per night: "))
    medical_conditions = int(input("Number of medical conditions: "))
    medications = int(input("Number of medications: "))
    hereditary_diseases = int(input("Number of hereditary diseases: "))
    mental_health = int(input("Rate your mental health (1-10): "))
    cholesterol_level = float(input("Enter your cholesterol level: "))
    blood_pressure = float(input("Enter your blood pressure: "))
    glucose_level = float(input("Enter your glucose level: "))
    activity_level = int(input("Rate your activity level (1-5): "))
    diet_type = int(input("Are you a vegetarian? (0 for no, 1 for yes): "))

    user_data = np.array([[
        height, weight, age, kids, gender, employment_status, marital_status, bmi, smoking, alcohol,
        exercise, diet, stress_level, sleep_hours, medical_conditions, medications, hereditary_diseases,
        mental_health, cholesterol_level, blood_pressure, glucose_level, activity_level, diet_type
    ]])
    
    return user_data

def predict_life_expectancy(model, scaler, user_data):
    user_data_scaled = scaler.transform(user_data)
    predicted_age = model.predict(user_data_scaled)
    return predicted_age[0][0]

# Load the trained model
model = load_model('src/model.keras')
# Load the original dataset for scaler fitting
dataset = loadtxt('src/data.txt', delimiter=',')
X = dataset[:, 0:23]
scaler = StandardScaler()
scaler.fit_transform(X)

# Get user input and predict life expectancy
user_data = get_user_input()
predicted_age = predict_life_expectancy(model, scaler, user_data)
print(f"Predicted age at death: {predicted_age:.2f} years")
