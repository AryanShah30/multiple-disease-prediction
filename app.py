import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import sklearn

# Load models and scalers
diabetes_model = pickle.load(open(
    "D:/Internship - Machine Learning/Projects/ML Project - Multiple Disease Prediction Model/saved models/diabetes_model.sav",
    "rb"))
diabetes_scaler = pickle.load(open(
    "D:/Internship - Machine Learning/Projects/ML Project - Multiple Disease Prediction Model/saved models/diabetes_scaler.sav",
    "rb"))
heart_model = pickle.load(open(
    "D:/Internship - Machine Learning/Projects/ML Project - Multiple Disease Prediction Model/saved models/heart_disease.sav",
    "rb"))
parkinsons_model = pickle.load(open(
    "D:/Internship - Machine Learning/Projects/ML Project - Multiple Disease Prediction Model/saved models/parkinssons_disease.sav",
    "rb"))
breast_cancer_model = pickle.load(open(
    "D:/Internship - Machine Learning/Projects/ML Project - Multiple Disease Prediction Model/saved models/breast_cancer_model.sav",
    "rb"))

# Page configuration
st.set_page_config(page_title="Disease Prediction", page_icon="⚕", layout="wide")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        [
            "Index",
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinson's Disease Prediction",
            "Breast Cancer Prediction",
        ],
        icons=["house", "activity", "heart", "person", "hospital"],
        default_index=0,
    )

# Index page
if selected == "Index":
    st.title("Multiple Disease Prediction System using Machine Learning")
    st.markdown("---")

    st.markdown("""
    This application leverages the power of machine learning to predict the likelihood of several health conditions based on user input. The diseases covered include Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer. By inputting relevant health metrics, users can receive a quick and accurate assessment of their health condition. The models have been trained on extensive datasets to ensure reliability and accuracy.

    ## Features:

    ### Diabetes Prediction
    Provide details such as the number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age to predict the likelihood of diabetes.

    ### Heart Disease Prediction
    Input your age, sex, chest pain types, resting blood pressure, serum cholesterol, fasting blood sugar levels, resting electrocardiograph results, maximum heart rate, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, major vessels colored by fluoroscopy, and thalassemia status to assess the risk of heart disease.

    ### Parkinson's Disease Prediction
    Enter parameters such as MDVP (Fo, Fhi, Flo), jitter, RAP, PPQ, DDP, shimmer, APQ, DDA, NHR, HNR, RPDE, DFA, spread, D2, and PPE to determine the probability of having Parkinson's disease.

    ### Breast Cancer Prediction
    Provide metrics including mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension, and their respective errors to predict if the breast cancer is benign or malignant.

    Each disease prediction model is carefully trained and validated to ensure high performance. We aim to provide a tool that helps users make informed decisions about their health.

    ## How to Use:
    1. Navigate through the sidebar to select the disease prediction model.
    2. Input the required health metrics.
    3. Click the button to receive the prediction result.
    4. The results will indicate whether the person is likely to have the disease or not.

    This application is designed to assist and provide preliminary insights. For comprehensive medical advice and diagnosis, please consult a healthcare professional.
    """)

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        SkinThickness = st.text_input("Skin Thickness Value")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")

    with col2:
        Glucose = st.text_input("Glucose Level")
        Insulin = st.text_input("Insulin Level")
        Age = st.text_input("Age of the Person")

    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
        BMI = st.text_input("BMI Value")

    diabetes_diagnosis = ""

    if st.button("Diabetes Test Result"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)
        user_input = diabetes_scaler.transform(user_input)
        diabetes_prediction = diabetes_model.predict(user_input)

        if diabetes_prediction[0] == 1:
            diabetes_diagnosis = "The person is diabetic"
        else:
            diabetes_diagnosis = "The person is not diabetic"

    st.success(diabetes_diagnosis)

# Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
        trestbps = st.text_input("Resting Blood Pressure")
        restecg = st.text_input("Resting Electrocardiograph results")
        oldpeak = st.text_input("ST depression induced by exercise")
        thal = st.text_input("thal: 1 = normal; 2 = fixed defect; 3 = reversable defect")

    with col2:
        sex = st.text_input("Sex")
        chol = st.text_input("Serum Cholesterol in mg/dl")
        thalach = st.text_input("Maximum Heart Rate achieved")
        slope = st.text_input("Slope of the peak exercise ST segment")

    with col3:
        cp = st.text_input("Chest Pain types")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl")
        exang = st.text_input("Exercise Induced Angina")
        ca = st.text_input("Major vessels colored by fluoroscopy")

    heart_diagnosis = ""

    if st.button("Heart Disease Test Result"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)
        heart_prediction = heart_model.predict(user_input)

        if heart_prediction[0] == 1:
            heart_diagnosis = "The person is having heart disease"
        else:
            heart_diagnosis = "The person does not have any heart disease"

    st.success(heart_diagnosis)

# Parkinson's Disease Prediction
if selected == "Parkinson's Disease Prediction":
    st.title("Parkinson's Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input("MDVP:Fo(Hz)")
        RAP = st.text_input("MDVP:RAP")
        APQ3 = st.text_input("Shimmer:APQ3")
        HNR = st.text_input("HNR")
        D2 = st.text_input("D2")

    with col2:
        fhi = st.text_input("MDVP:Fhi(Hz)")
        PPQ = st.text_input("MDVP:PPQ")
        APQ5 = st.text_input("Shimmer:APQ5")
        RPDE = st.text_input("RPDE")
        PPE = st.text_input("PPE")

    with col3:
        flo = st.text_input("MDVP:Flo(Hz)")
        DDP = st.text_input("Jitter:DDP")
        APQ = st.text_input("MDVP:APQ")
        DFA = st.text_input("DFA")

    with col4:
        Jitter_percent = st.text_input("MDVP:Jitter(%)")
        Shimmer = st.text_input("MDVP:Shimmer")
        DDA = st.text_input("Shimmer:DDA")
        spread1 = st.text_input("spread1")

    with col5:
        Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
        Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
        NHR = st.text_input("NHR")
        spread2 = st.text_input("spread2")

    parkinsons_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ,
                      DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)
        parkinsons_prediction = parkinsons_model.predict(user_input)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# Breast Cancer Prediction
if selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mr = st.text_input("Mean Radius")
        mc = st.text_input("Mean Compactness")
        re = st.text_input("Radius Error")
        coerr = st.text_input("Compactness Error")
        wr = st.text_input("Worst Radius")
        wcom = st.text_input("Worst Compactness")

    with col2:
        mt = st.text_input("Mean Texture")
        mcon = st.text_input("Mean Concavity")
        te = st.text_input("Texture Error")
        conerr = st.text_input("Concavity Error")
        wt = st.text_input("Worst Texture")
        wcon = st.text_input("Worst Concavity")

    with col3:
        mp = st.text_input("Mean Perimeter")
        mconpt = st.text_input("Mean Concave Points")
        pe = st.text_input("Perimeter Error")
        conpterr = st.text_input("Concave Points Error")
        wp = st.text_input("Worst Perimeter")
        wconpt = st.text_input("Worst Concave Points")

    with col4:
        ma = st.text_input("Mean Area")
        msym = st.text_input("Mean Symmetry")
        ae = st.text_input("Area Error")
        symerr = st.text_input("Symmetry Error")
        wa = st.text_input("Worst Area")
        wsym = st.text_input("Worst Symmetry")

    with col5:
        ms = st.text_input("Mean Smoothness")
        mf = st.text_input("Mean Fractional Dimension")
        se = st.text_input("Smoothness Error")
        frdierr = st.text_input("Fractional Dimension Error")
        ws = st.text_input("Worst Smoothness")
        wfd = st.text_input("Worst Fractional Dimension")

    breast_cancer_diagnosis = ""

    if st.button("Breast Cancer Test Result"):
        user_input = [mr, mt, mp, ma, ms, mc, mcon, mconpt, msym, mf, re, te, pe, ae, se, coerr, conerr, conpterr,
                      symerr, frdierr, wr, wt, wp, wa, ws, wcom, wcon, wconpt, wsym, wfd]
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)
        breast_cancer_prediction = breast_cancer_model.predict(user_input)

        if breast_cancer_prediction[0] == 1:
            breast_cancer_diagnosis = "The Breast Cancer is Benign"
        else:
            breast_cancer_diagnosis = "The Breast Cancer is Malignant"

    st.success(breast_cancer_diagnosis)
