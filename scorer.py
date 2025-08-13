import joblib
import pandas as pd

def predict_enrollment(applicant_data):
    """
    Loads the saved model and predicts enrollment for a new applicant.
    
    Args:
        applicant_data (dict): A dictionary containing the applicant's data.
        
    Returns:
        dict: A dictionary with the prediction and probability.
    """
    try:
        # Load the trained model from the file
        model = joblib.load('enrollment_model.joblib')
    except FileNotFoundError:
        return {"error": "Model file not found. Make sure 'enrollment_model.joblib' is in the same directory."}

    # Define the exact feature order the model was trained on
    feature_order = [
        'gpa', 'campus_visit', 'contacted_by_recruiter', 
        'distance_from_home_km', 'reapplicant', 'scholarship_offered'
    ]

    # Convert the input dictionary to a pandas DataFrame
    applicant_df = pd.DataFrame([applicant_data])
    
    # Ensure the columns are in the correct order
    applicant_df = applicant_df[feature_order]

    # Make predictions
    prediction_val = model.predict(applicant_df)[0]
    probability_val = model.predict_proba(applicant_df)[0][1] # Probability of class '1'

    result = {
        "prediction": "Enrolled" if prediction_val == 1 else "Not Enrolled",
        "enrollment_probability": f"{probability_val:.2%}"
    }
    
    return result