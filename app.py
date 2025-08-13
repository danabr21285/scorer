Of course. Here is the complete and final set of code for the entire project, structured into the four necessary files.

File 1: train_model.py
This script generates the data, finds the best model settings using GridSearchCV, and saves the final, trained model to the file enrollment_model.joblib. You only need to run this script once to create your model file.

Python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Step 1: Generate the Data ---
num_students = 5000
np.random.seed(42)

data = {
    'gpa': np.random.normal(loc=3.3, scale=0.4, size=num_students).clip(1.0, 4.0),
    'campus_visit': np.random.choice([1, 0], size=num_students, p=[0.45, 0.55]),
    'contacted_by_recruiter': np.random.choice([1, 0], size=num_students, p=[0.6, 0.4]),
    'distance_from_home_km': (np.random.exponential(scale=300, size=num_students) + 10).astype(int),
    'reapplicant': np.random.choice([1, 0], size=num_students, p=[0.08, 0.92]),
    'scholarship_offered': np.random.choice([1, 0], size=num_students, p=[0.3, 0.7]),
}
df = pd.DataFrame(data)

# Create the target variable based on the features
enrollment_prob = np.full(num_students, 0.10)
enrollment_prob += (df['gpa'] - 3.3) * 0.20
enrollment_prob[df['campus_visit'] == 1] += 0.25
enrollment_prob[df['contacted_by_recruiter'] == 1] += 0.05
enrollment_prob -= df['distance_from_home_km'] / 5000
enrollment_prob[df['reapplicant'] == 1] += 0.20
enrollment_prob[df['scholarship_offered'] == 1] += 0.35
enrollment_prob = enrollment_prob.clip(0, 1)
df['enrolled'] = (enrollment_prob > np.random.rand(num_students)).astype(int)

# Define features (X) and target (y)
features = ['gpa', 'campus_visit', 'contacted_by_recruiter', 
            'distance_from_home_km', 'reapplicant', 'scholarship_offered']
X = df[features]
y = df['enrolled']

# Split data for training
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- Step 2: Find the Best Model with GridSearchCV ---
print("Starting hyperparameter tuning to find the best model...")
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': ['balanced', 'balanced_subsample']
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='recall', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")


# --- Step 3: Save the Final Model ---
filename = 'enrollment_model.joblib'
joblib.dump(best_model, filename)
print(f"\nTraining complete. Model saved to '{filename}'.")
