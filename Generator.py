#Author : t.me/Jelofen1962
#MIT Copy-Right, you can use this model but just With the CREDIT


import numpy as np
import random

def generate_realistic_data(num_samples):
    data = []
    for _ in range(num_samples):
        height = np.random.normal(168, 10)
        weight = np.random.normal(70, 15)
        age = np.random.randint(18, 90)
        kids = np.random.randint(0, 5) if age > 30 else 0
        gender = random.choice([0, 1])
        employment_status = random.choice([0, 1])
        marital_status = random.choice([0, 1])
        bmi = weight / ((height / 100) ** 2)
        smoking = random.choice([0, 1])
        alcohol = random.choice([0, 1])
        exercise = random.choice([0, 1])
        diet = random.choice([0, 1])
        stress_level = np.random.randint(1, 10)
        sleep_hours = np.random.normal(7, 1.5)
        medical_conditions = np.random.randint(0, 5)
        medications = np.random.randint(0, 3)
        hereditary_diseases = np.random.randint(0, 3)
        mental_health = np.random.randint(1, 10)
        cholesterol_level = np.random.normal(200, 30)
        blood_pressure = np.random.normal(120, 15)
        glucose_level = np.random.normal(90, 10)
        activity_level = np.random.randint(1, 5)
        diet_type = random.choice([0, 1])
        live_age = age + np.random.normal(20, 10)

        # Round to 5 decimal places
        data.append([
            round(height, 5), round(weight, 5), age, kids, gender, employment_status, marital_status, round(bmi, 5),
            smoking, alcohol, exercise, diet, stress_level, round(sleep_hours, 5), medical_conditions, medications,
            hereditary_diseases, mental_health, round(cholesterol_level, 5), round(blood_pressure, 5), round(glucose_level, 5),
            activity_level, diet_type, round(live_age, 5)
        ])
    return np.array(data)

def save_data_with_fixed_precision(filename, data, precision=8):
    with open(filename, 'w') as file:
        for row in data:
            formatted_row = ','.join([f"{x:.{precision}f}" if isinstance(x, float) else str(x) for x in row])
            file.write(formatted_row + '\n')

# Generate dataset
num_samples = 200
dataset = generate_realistic_data(num_samples)
save_data_with_fixed_precision('src/data.txt', dataset)
