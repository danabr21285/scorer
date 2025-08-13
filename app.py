import streamlit as st
from scorer import predict_enrollment

st.title('Student Enrollment Predictor')
st.write("Enter the applicant's details to get a prediction.")

# Create Input Fields for Each Feature
gpa = st.slider('GPA', 1.0, 4.0, 3.5)
campus_visit = st.selectbox('Visited Campus?', ('Yes', 'No'))
contacted_by_recruiter = st.selectbox('Contacted by Recruiter?', ('Yes', 'No'))
distance_from_home_km = st.number_input('Distance from Home (km)', 0, value=150)
reapplicant = st.selectbox('Is this a Reapplicant?', ('Yes', 'No'))
scholarship_offered = st.selectbox('Was a Scholarship Offered?', ('Yes', 'No'))

# Submit Button
if st.button('Predict Enrollment'):
    applicant_data = {
        'gpa': gpa,
        'campus_visit': 1 if campus_visit == 'Yes' else 0,
        'contacted_by_recruiter': 1 if contacted_by_recruiter == 'Yes' else 0,
        'distance_from_home_km': distance_from_home_km,
        'reapplicant': 1 if reapplicant == 'Yes' else 0,
        'scholarship_offered': 1 if scholarship_offered == 'Yes' else 0
    }
    
    result = predict_enrollment(applicant_data)
    
    st.subheader('Prediction Result')
    st.write(f"**Prediction:** {result['prediction']}")
    st.write(f"**Confidence Score:** {result['enrollment_probability']}")


