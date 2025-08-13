import joblib
import pandas as pd

def predict_enrollment(applicant_data):
    """
    Loads the saved model and predicts enrollment for a new applicant.
    """
    try:
        model = joblib.load('enrollment_model.joblib')
    except FileNotFoundError:
        return {"error": "Model file not found. Make sure 'enrollment_model.joblib' is in the same directory."}

    # Define the exact feature order the model expects
    feature_order = [
        'gpa', 'campus_visit', 'contacted_by_recruiter', 
        'distance_from_home_km', 'reapplicant', 'scholarship_offered'
    ]

    # Convert input data to a DataFrame with the correct column order
    applicant_df = pd.DataFrame([applicant_data])
    applicant_df = applicant_df[feature_order]

    # Make predictions
    prediction_val = model.predict(applicant_df)[0]
    probability_val = model.predict_proba(applicant_df)[0][1]

    result = {
        "prediction": "Enrolled" if prediction_val == 1 else "Not Enrolled",
        "enrollment_probability": f"{probability_val:.2%}"
    }
    
    return result
