import os
import pickle
import streamlit as st
import subprocess
import sys

# Try to import streamlit_option_menu, install if not available
try:
    from streamlit_option_menu import option_menu
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-option-menu"])
    from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
try:
    diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
st.sidebar.header("Made By Subhadip 🔥")
# sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )



# Helper function to check if model supports predict_proba
def has_predict_proba(model):
    return hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba', None))

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML 🩸')
    
    # Information about input values
    with st.expander("ℹ️ About Input Values"):
        st.write("""
        - **Pregnancies**: Number of times pregnant
        - **Glucose**: Plasma glucose concentration (mg/dL)
        - **Blood Pressure**: Diastolic blood pressure (mm Hg)
        - **Skin Thickness**: Triceps skin fold thickness (mm)
        - **Insulin**: 2-Hour serum insulin (mu U/mL)
        - **BMI**: Body mass index (weight in kg/(height in m)^2)
        - **Diabetes Pedigree Function**: Diabetes family history function
        - **Age**: Age in years
        """)

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)

    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0.0, max_value=200.0, value=80.0, step=0.1)

    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0.0, max_value=150.0, value=80.0, step=0.1)

    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0.0, max_value=100.0, value=20.0, step=0.1)

    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0.0, max_value=850.0, value=80.0, step=0.1)

    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=25.0, step=0.1)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=3.0, value=0.5, step=0.01)

    with col2:
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=25, step=1)

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
                st.error(diab_diagnosis)
            else:
                diab_diagnosis = 'The person is not diabetic'
                st.success(diab_diagnosis)
                
            # Show probability score if available
            if has_predict_proba(diabetes_model):
                probability = diabetes_model.predict_proba([user_input])
                st.info(f"Probability of being diabetic: {probability[0][1]*100:.2f}%")
            else:
                st.info("Probability scores not available for this model")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    
    # Information about input values
    with st.expander("ℹ️ About Input Values"):
        st.write("""
        - **Age**: Age in years
        - **Sex**: Gender (1 = male; 0 = female)
        - **Chest Pain Type**: 
            - 0: Typical angina
            - 1: Atypical angina
            - 2: Non-anginal pain
            - 3: Asymptomatic
        - **Resting Blood Pressure**: in mm Hg on admission
        - **Serum Cholesterol**: in mg/dl
        - **Fasting Blood Sugar**: > 120 mg/dl (1 = true; 0 = false)
        - **Resting ECG Results**:
            - 0: Normal
            - 1: ST-T wave abnormality
            - 2: Probable or definite left ventricular hypertrophy
        - **Max Heart Rate**: Maximum heart rate achieved
        - **Exercise Induced Angina**: (1 = yes; 0 = no)
        - **ST Depression**: induced by exercise relative to rest
        - **Slope**: Slope of the peak exercise ST segment
        - **Major Vessels**: Number of major vessels colored by fluoroscopy (0-3)
        - **Thal**: 
            - 0: Normal
            - 1: Fixed defect
            - 2: Reversable defect
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50, step=1)

    with col2:
        sex = st.selectbox('Sex', options=[('Female', 0), ('Male', 1)], format_func=lambda x: x[0])[1]

    with col3:
        cp = st.selectbox('Chest Pain types', 
                         options=[('Typical Angina', 0), ('Atypical Angina', 1), 
                                 ('Non-anginal Pain', 2), ('Asymptomatic', 3)],
                         format_func=lambda x: x[0])[1]

    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120, step=1)

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=200, step=1)

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[('False', 0), ('True', 1)], format_func=lambda x: x[0])[1]

    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', 
                              options=[('Normal', 0), ('ST-T wave abnormality', 1), 
                                      ('Left ventricular hypertrophy', 2)],
                              format_func=lambda x: x[0])[1]

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, max_value=250, value=150, step=1)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', 
                           options=[('Upsloping', 0), ('Flat', 1), ('Downsloping', 2)],
                           format_func=lambda x: x[0])[1]

    with col3:
        ca = st.slider('Major vessels colored by flourosopy', min_value=0, max_value=3, value=0)

    with col1:
        thal = st.selectbox('Thalassemia', 
                          options=[('Normal', 0), ('Fixed defect', 1), ('Reversable defect', 2)],
                          format_func=lambda x: x[0])[1]

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
                st.error(heart_diagnosis)
            else:
                heart_diagnosis = 'The person does not have any heart disease'
                st.success(heart_diagnosis)
                
            # Show probability score if available
            if has_predict_proba(heart_disease_model):
                probability = heart_disease_model.predict_proba([user_input])
                st.info(f"Probability of having heart disease: {probability[0][1]*100:.2f}%")
            else:
                st.info("Probability scores not available for this model")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    
    # Information about input values
    with st.expander("ℹ️ About Input Values"):
        st.write("""
        These are voice measurement parameters from Parkinson's patients and healthy individuals.
        Most are MDVP (Multi-Dimensional Voice Program) parameters that measure various aspects of vocal oscillations.
        """)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=80.0, max_value=260.0, value=145.0, step=0.1)

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=100.0, max_value=600.0, value=200.0, step=0.1)

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=60.0, max_value=250.0, value=150.0, step=0.1)

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.4f")

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.0003, step=0.00001, format="%.5f")

    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=0.1, value=0.003, step=0.0001, format="%.4f")

    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=0.1, value=0.003, step=0.0001, format="%.4f")

    with col3:
        DDP = st.number_input('Jitter:DDP', min_value=0.0, max_value=0.1, value=0.01, step=0.0001, format="%.4f")

    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=0.2, value=0.02, step=0.0001, format="%.4f")

    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=5.0, value=0.2, step=0.01, format="%.2f")

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=0.2, value=0.01, step=0.0001, format="%.4f")

    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=0.2, value=0.01, step=0.0001, format="%.4f")

    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, max_value=0.2, value=0.01, step=0.0001, format="%.4f")

    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, max_value=0.2, value=0.01, step=0.0001, format="%.4f")

    with col5:
        NHR = st.number_input('NHR', min_value=0.0, max_value=1.0, value=0.01, step=0.0001, format="%.4f")

    with col1:
        HNR = st.number_input('HNR', min_value=0.0, max_value=40.0, value=20.0, step=0.1)

    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")

    with col3:
        DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.7, step=0.0001, format="%.4f")

    with col4:
        spread1 = st.number_input('spread1', min_value=-10.0, max_value=0.0, value=-5.0, step=0.01)

    with col5:
        spread2 = st.number_input('spread2', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    with col1:
        D2 = st.number_input('D2', min_value=0.0, max_value=10.0, value=2.0, step=0.01)

    with col2:
        PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
                st.error(parkinsons_diagnosis)
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
                st.success(parkinsons_diagnosis)
                
            # Show probability score if available
            if has_predict_proba(parkinsons_model):
                probability = parkinsons_model.predict_proba([user_input])
                st.info(f"Probability of having Parkinson's disease: {probability[0][1]*100:.2f}%")
            else:
                st.info("Probability scores not available for this model")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")